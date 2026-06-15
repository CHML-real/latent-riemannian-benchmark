from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from lrbench.decoders.toy_decoder import gaussian_bump_decoder
from lrbench.methods.euclidean_line import euclidean_line
from lrbench.methods.lrw_bvp_diagnostic import (
    aggregate_diagnostic_cases,
    compact_case_metrics,
    extract_solver_output,
    summarize_bvp_path,
)
from lrbench.methods.lrw_bvp_solver import get_lrw_bvp_classes, normalize_lrw_path
from lrbench.metrics.pullback_energy import pullback_path_energy, pullback_path_length
from lrbench.utils.device import resolve_device, resolve_dtype
from lrbench.utils.io import load_yaml, save_json


def _as_list(value: Any, default: list[Any]) -> list[Any]:
    if value is None:
        return default
    if isinstance(value, list):
        return value
    return [value]


def _make_endpoint_pair(left_x: float, right_x: float, base_y: float, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    z0 = torch.tensor([[left_x, base_y]], device=device, dtype=dtype)
    z1 = torch.tensor([[right_x, base_y]], device=device, dtype=dtype)
    return z0, z1


def _baseline(z0: torch.Tensor, z1: torch.Tensor, n_points: int, beta: float, sigma: float, cross: float, regularization: float) -> dict[str, Any]:
    straight = euclidean_line(z0, z1, num_points=n_points)
    energy = pullback_path_energy(straight, beta=beta, sigma=sigma, cross=cross, damping=regularization)
    length = pullback_path_length(straight, beta=beta, sigma=sigma, cross=cross, damping=regularization)
    return {
        "straight_pullback_energy_mean": energy["pullback_energy_mean"],
        "straight_pullback_length_mean": length["pullback_length_mean"],
    }


def _call_solver_method(
    solver: Any,
    method_name: str,
    z0: torch.Tensor,
    z1: torch.Tensor,
    n_points: int,
) -> tuple[Any, dict[str, Any]]:
    if method_name == "geodesic_path":
        return extract_solver_output(solver.geodesic_path(z0, z1, n_points=n_points))
    if method_name == "solve":
        return extract_solver_output(solver.solve(z0, z1))
    raise ValueError(f"unsupported method_name: {method_name}")


def _run_single_case(
    *,
    case_id: str,
    method_name: str,
    bvp_cls: type,
    metric_cls: type,
    z0: torch.Tensor,
    z1: torch.Tensor,
    decoder: Any,
    n_points: int,
    regularization: float,
    chunk_size: int | None,
    solver_n_steps: int,
    solver_step_size: float,
    solver_lr: float,
    solver_max_iter: int,
    solver_tol: float,
    endpoint_tolerance: float,
    beta: float,
    sigma: float,
    cross: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        metric = metric_cls(decoder=decoder, chunk_size=chunk_size, regularization=regularization)
        solver = bvp_cls(metric=metric, n_steps=solver_n_steps, step_size=solver_step_size, lr=solver_lr, max_iter=solver_max_iter, tol=solver_tol)
        raw_path, solver_info = _call_solver_method(solver, method_name, z0, z1, n_points)
        path = normalize_lrw_path(raw_path, z0, z1, requested_n_points=n_points)
        diagnostics = summarize_bvp_path(path, z0, z1, endpoint_tolerance, beta=beta, sigma=sigma, cross=cross, regularization=regularization)
        baseline = _baseline(z0, z1, path.shape[0], beta, sigma, cross, regularization)
        diagnostics["raw"]["lrw_bvp_energy_ratio_over_straight"] = diagnostics["raw"].get("lrw_bvp_pullback_energy_mean", 0.0) / max(baseline["straight_pullback_energy_mean"], 1e-12)
        diagnostics["clamped"]["clamped_lrw_bvp_energy_ratio_over_straight"] = diagnostics["clamped"].get("clamped_lrw_bvp_pullback_energy_mean", 0.0) / max(baseline["straight_pullback_energy_mean"], 1e-12)
        status = "success"
        failure = {"has_failure": False, "failure_type": None, "message": None}
        raw_path_shape = list(raw_path.shape) if isinstance(raw_path, torch.Tensor) else None
    except Exception as exc:  # pragma: no cover - env dependent
        diagnostics = {}
        baseline = {}
        solver_info = {}
        raw_path_shape = None
        status = "failure"
        failure = {"has_failure": True, "failure_type": "LRW_F02_BVP_CALL_FAILED", "message": str(exc)}
    runtime_ms = (time.perf_counter() - start) * 1000.0
    result = {
        "case_id": case_id,
        "method_name": method_name,
        "status": status,
        "params": {
            "solver_n_steps": solver_n_steps,
            "solver_step_size": solver_step_size,
            "solver_lr": solver_lr,
            "solver_max_iter": solver_max_iter,
            "solver_tol": solver_tol,
            "n_points": n_points,
        },
        "raw_path_shape": raw_path_shape,
        "solver_info": solver_info,
        "baseline": baseline,
        "diagnostics": diagnostics,
        "compact": {},
        "runtime_ms": float(runtime_ms),
        "failure": failure,
    }
    result["compact"] = compact_case_metrics(result)
    return result


def run(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    device = resolve_device(str(cfg.get("device", "auto")))
    dtype = resolve_dtype(str(cfg.get("dtype", "float32")))
    left_x = float(cfg.get("left_x", -2.0))
    right_x = float(cfg.get("right_x", 2.0))
    base_y = float(cfg.get("base_y", 0.0))
    beta = float(cfg.get("beta", 6.0))
    sigma = float(cfg.get("sigma", 0.7))
    cross = float(cfg.get("cross", 0.2))
    regularization = float(cfg.get("regularization", cfg.get("damping", 1e-5)))
    endpoint_tolerance = float(cfg.get("endpoint_tolerance", 1e-3))
    chunk_size_value = cfg.get("chunk_size", None)
    chunk_size = None if chunk_size_value in (None, "null", "None") else int(chunk_size_value)

    methods = [str(x) for x in _as_list(cfg.get("methods"), ["geodesic_path", "solve"])]
    n_points_values = [int(x) for x in _as_list(cfg.get("n_points"), [24])]
    n_steps_values = [int(x) for x in _as_list(cfg.get("solver_n_steps"), [20])]
    lr_values = [float(x) for x in _as_list(cfg.get("solver_lr"), [0.1])]
    max_iter_values = [int(x) for x in _as_list(cfg.get("solver_max_iter"), [30])]
    tol_values = [float(x) for x in _as_list(cfg.get("solver_tol"), [1e-3])]
    step_size = float(cfg.get("solver_step_size", 0.05))

    bvp_cls, metric_cls, availability = get_lrw_bvp_classes()

    base_result = {
        "benchmark_id": cfg.get("benchmark_id", "lrw_bvp_diagnostic_001"),
        "benchmark_layer": "adapter",
        "target": "lrw_bvp_solver",
        "manifold": "toy_decoder_pullback_metric",
        "method": "lrw_bvp_diagnostic",
        "seed": seed,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "latent_dimension": 2,
        "decoder_output_dimension": 4,
        "regularization": regularization,
        "chunk_size": chunk_size,
        "endpoint_tolerance": endpoint_tolerance,
        "beta": beta,
        "sigma": sigma,
        "cross": cross,
        "lrw": {
            "available": availability.available,
            "bvp_class_path": availability.bvp_class_path,
            "metric_class_path": availability.metric_class_path,
            "reason": availability.reason,
        },
    }

    if bvp_cls is None or metric_cls is None:
        result = {
            **base_result,
            "status": "skipped",
            "cases": [],
            "summary": {},
            "metrics": {"available": False, "runtime_ms": 0.0},
            "failure": {"has_failure": False, "failure_type": None, "message": None},
            "skip": {"is_skipped": True, "reason": availability.reason or "LRW unavailable"},
        }
    else:
        z0, z1 = _make_endpoint_pair(left_x, right_x, base_y, device, dtype)

        def decoder(x: torch.Tensor) -> torch.Tensor:
            return gaussian_bump_decoder(x, beta=beta, sigma=sigma, cross=cross)

        all_start = time.perf_counter()
        cases: list[dict[str, Any]] = []
        case_idx = 0
        for method_name in methods:
            for n_points in n_points_values:
                for n_steps in n_steps_values:
                    for lr in lr_values:
                        for max_iter in max_iter_values:
                            for tol in tol_values:
                                case_idx += 1
                                cases.append(
                                    _run_single_case(
                                        case_id=f"case_{case_idx:03d}",
                                        method_name=method_name,
                                        bvp_cls=bvp_cls,
                                        metric_cls=metric_cls,
                                        z0=z0,
                                        z1=z1,
                                        decoder=decoder,
                                        n_points=n_points,
                                        regularization=regularization,
                                        chunk_size=chunk_size,
                                        solver_n_steps=n_steps,
                                        solver_step_size=step_size,
                                        solver_lr=lr,
                                        solver_max_iter=max_iter,
                                        solver_tol=tol,
                                        endpoint_tolerance=endpoint_tolerance,
                                        beta=beta,
                                        sigma=sigma,
                                        cross=cross,
                                    )
                                )
        runtime_ms = (time.perf_counter() - all_start) * 1000.0
        summary = aggregate_diagnostic_cases(cases)
        # Diagnostic runner succeeds if it completes; endpoint misses are findings, not runner failures.
        result = {
            **base_result,
            "status": "success",
            "cases": cases,
            "summary": summary,
            "metrics": {
                "available": True,
                "runtime_ms": float(runtime_ms),
                "diagnostic_case_count": summary.get("case_count", 0),
                "diagnostic_call_failure_rate": summary.get("call_failure_rate", 0.0),
                "diagnostic_raw_endpoint_failure_rate": summary.get("raw_endpoint_failure_rate", 0.0),
                "diagnostic_raw_endpoint_pass_count": summary.get("raw_endpoint_pass_count", 0),
                "diagnostic_clamped_endpoint_pass_count": summary.get("clamped_endpoint_pass_count", 0),
                "diagnostic_raw_endpoint_error_mean_min": summary.get("raw_endpoint_error_mean_min"),
                "diagnostic_raw_endpoint_error_mean_max": summary.get("raw_endpoint_error_mean_max"),
                "diagnostic_energy_improved_count": summary.get("energy_improved_count", 0),
                "diagnostic_energy_only_success_count": summary.get("energy_only_success_count", 0),
                "diagnostic_overall_valid_count": summary.get("overall_valid_count", 0),
                "diagnostic_overall_valid_rate": summary.get("overall_valid_rate", 0.0),
                "diagnostic_endpoint_tolerance_pass_rate": summary.get("endpoint_tolerance_pass_rate", 0.0),
                "diagnostic_boundary_condition_failure_count": summary.get("boundary_condition_failure_count", 0),
                "diagnostic_best_raw_endpoint_error": (summary.get("best_by_endpoint") or {}).get("raw_endpoint_error_mean"),
                "diagnostic_best_raw_energy_ratio": (summary.get("best_by_energy") or {}).get("raw_energy_ratio_over_straight"),
                "diagnostic_best_valid_score": (summary.get("best_by_valid_score") or {}).get("valid_geodesic_score"),
            },
            "failure": {"has_failure": False, "failure_type": None, "message": None},
            "skip": {"is_skipped": False, "reason": None},
        }

    output_dir = Path(cfg.get("output_dir", "results/raw"))
    output_path = output_dir / f"{result['benchmark_id']}.json"
    save_json(result, output_path)
    print(f"saved: {output_path}")
    print(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LRW BVPSolver diagnostic benchmark")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
