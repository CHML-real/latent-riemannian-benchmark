from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from lrbench.decoders.toy_decoder import gaussian_bump_decoder
from lrbench.methods.euclidean_line import euclidean_line
from lrbench.methods.lrw_bvp_solver import (
    compute_lrw_bvp_geodesic_path,
    lrw_bvp_path_metrics,
    normalize_lrw_path,
)
from lrbench.methods.pullback_paths import pullback_bump_detour_path
from lrbench.metrics.accuracy import endpoint_error
from lrbench.metrics.pullback_energy import pullback_path_energy, pullback_path_length
from lrbench.utils.device import resolve_device, resolve_dtype
from lrbench.utils.io import load_yaml, save_json


def _make_endpoint_pairs(
    batch_size: int,
    left_x: float,
    right_x: float,
    base_y: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    z0 = torch.empty(batch_size, 2, device=device, dtype=dtype)
    z1 = torch.empty(batch_size, 2, device=device, dtype=dtype)
    z0[:, 0] = left_x
    z1[:, 0] = right_x
    z0[:, 1] = base_y
    z1[:, 1] = base_y
    return z0, z1


def _baseline_metrics(
    z0_batch: torch.Tensor,
    z1_batch: torch.Tensor,
    num_points: int,
    detour_y: float,
    beta: float,
    sigma: float,
    cross: float,
    regularization: float,
) -> dict[str, Any]:
    straight = euclidean_line(z0_batch, z1_batch, num_points=num_points)
    detour = pullback_bump_detour_path(z0_batch, z1_batch, detour_y=detour_y, num_points=num_points)
    metrics: dict[str, Any] = {}
    for label, path in [("straight", straight), ("detour", detour)]:
        energy = pullback_path_energy(path, beta=beta, sigma=sigma, cross=cross, damping=regularization)
        length = pullback_path_length(path, beta=beta, sigma=sigma, cross=cross, damping=regularization)
        metrics[f"{label}_pullback_energy_mean"] = energy["pullback_energy_mean"]
        metrics[f"{label}_pullback_length_mean"] = length["pullback_length_mean"]
    return metrics


def run(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    device = resolve_device(str(cfg.get("device", "auto")))
    dtype = resolve_dtype(str(cfg.get("dtype", "float32")))
    batch_size = int(cfg.get("batch_size", 1))
    n_points = int(cfg.get("n_points", cfg.get("num_points", 32)))
    left_x = float(cfg.get("left_x", -2.0))
    right_x = float(cfg.get("right_x", 2.0))
    base_y = float(cfg.get("base_y", 0.0))
    detour_y = float(cfg.get("detour_y", 1.5))
    beta = float(cfg.get("beta", 6.0))
    sigma = float(cfg.get("sigma", 0.7))
    cross = float(cfg.get("cross", 0.2))
    regularization = float(cfg.get("regularization", cfg.get("damping", 1e-5)))
    chunk_size_value = cfg.get("chunk_size", None)
    chunk_size = None if chunk_size_value in (None, "null", "None") else int(chunk_size_value)

    solver_n_steps = int(cfg.get("solver_n_steps", 20))
    solver_step_size = float(cfg.get("solver_step_size", 0.05))
    solver_lr = float(cfg.get("solver_lr", 0.1))
    solver_max_iter = int(cfg.get("solver_max_iter", 50))
    solver_tol = float(cfg.get("solver_tol", 1e-3))
    endpoint_tolerance = float(cfg.get("endpoint_tolerance", 1e-3))

    z0_batch, z1_batch = _make_endpoint_pairs(batch_size, left_x, right_x, base_y, device, dtype)

    def decoder(x: torch.Tensor) -> torch.Tensor:
        return gaussian_bump_decoder(x, beta=beta, sigma=sigma, cross=cross)

    # LRW BVPSolver expects batched endpoint tensors shaped [B, D].
    # Keep a batch axis even when benchmarking a single endpoint pair.
    # The previous v1.0 version squeezed this to [D], which can trigger
    # `not enough values to unpack (expected 2, got 1)` inside LRW.
    use_single_pair = bool(cfg.get("use_single_pair", True))
    call_z0 = z0_batch[:1] if use_single_pair else z0_batch
    call_z1 = z1_batch[:1] if use_single_pair else z1_batch
    metric_z0 = call_z0
    metric_z1 = call_z1

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    try:
        raw_path, lrw_info, solver_info = compute_lrw_bvp_geodesic_path(
            call_z0,
            call_z1,
            decoder=decoder,
            regularization=regularization,
            chunk_size=chunk_size,
            solver_n_steps=solver_n_steps,
            solver_step_size=solver_step_size,
            solver_lr=solver_lr,
            solver_max_iter=solver_max_iter,
            solver_tol=solver_tol,
            n_points=n_points,
        )
        exception_message = None
    except Exception as exc:  # pragma: no cover - exact LRW failures are env-dependent
        raw_path, lrw_info, solver_info = None, {"available": True, "reason": str(exc)}, {}
        exception_message = str(exc)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    base_result = {
        "benchmark_id": cfg.get("benchmark_id", "lrw_bvp_001"),
        "benchmark_layer": "adapter",
        "target": "lrw_bvp_solver",
        "manifold": "toy_decoder_pullback_metric",
        "method": "lrw_bvp_geodesic_path",
        "seed": seed,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "latent_dimension": 2,
        "decoder_output_dimension": 4,
        "batch_size": 1 if use_single_pair else batch_size,
        "n_points_requested": n_points,
        "regularization": regularization,
        "chunk_size": chunk_size,
        "solver_n_steps": solver_n_steps,
        "solver_step_size": solver_step_size,
        "solver_lr": solver_lr,
        "solver_max_iter": solver_max_iter,
        "solver_tol": solver_tol,
        "beta": beta,
        "sigma": sigma,
        "cross": cross,
    }

    if raw_path is None:
        reason = exception_message or lrw_info.get("reason") or "LRW BVPSolver unavailable"
        result = {
            **base_result,
            "status": "skipped" if not lrw_info.get("available", False) else "failure",
            "lrw": lrw_info,
            "solver_info": solver_info,
            "metrics": {
                "available": bool(lrw_info.get("available", False)),
                "runtime_ms": float(runtime_ms),
                "peak_memory_mb": float(peak_memory_mb),
            },
            "failure": {
                "has_failure": bool(lrw_info.get("available", False)),
                "failure_type": "LRW_F02_BVP_CALL_FAILED" if lrw_info.get("available", False) else None,
                "message": reason if lrw_info.get("available", False) else None,
            },
            "skip": {
                "is_skipped": not bool(lrw_info.get("available", False)),
                "reason": reason if not lrw_info.get("available", False) else None,
            },
        }
    else:
        path = normalize_lrw_path(raw_path, metric_z0, metric_z1, requested_n_points=n_points)
        metrics: dict[str, Any] = {}
        metrics.update(lrw_bvp_path_metrics(path, metric_z0, metric_z1, beta=beta, sigma=sigma, cross=cross, regularization=regularization))
        metrics.update(_baseline_metrics(metric_z0, metric_z1, n_points, detour_y, beta, sigma, cross, regularization))
        metrics["available"] = True
        metrics["runtime_ms"] = float(runtime_ms)
        metrics["peak_memory_mb"] = float(peak_memory_mb)
        metrics["solver_info_key_count"] = len(solver_info)
        metrics["lrw_bvp_energy_ratio_over_straight"] = metrics["lrw_bvp_pullback_energy_mean"] / max(metrics["straight_pullback_energy_mean"], 1e-12)
        metrics["lrw_bvp_length_ratio_over_straight"] = metrics["lrw_bvp_pullback_length_mean"] / max(metrics["straight_pullback_length_mean"], 1e-12)

        has_nan_inf = bool(metrics["lrw_bvp_has_nan"] or metrics["lrw_bvp_has_inf"])
        endpoint_failed = bool(metrics["lrw_bvp_endpoint_error_mean"] > endpoint_tolerance)
        has_failure = has_nan_inf or endpoint_failed
        if has_nan_inf:
            failure_type = "GEO_F06_NAN_INF"
            message = "NaN or Inf detected in LRW BVP geodesic path"
        elif endpoint_failed:
            failure_type = "GEO_F01_ENDPOINT_MISS"
            message = "LRW BVP path endpoints deviate beyond tolerance"
        else:
            failure_type = None
            message = None

        result = {
            **base_result,
            "status": "failure" if has_failure else "success",
            "lrw": lrw_info,
            "solver_info": solver_info,
            "metrics": metrics,
            "failure": {"has_failure": has_failure, "failure_type": failure_type, "message": message},
            "skip": {"is_skipped": False, "reason": None},
        }

    output_dir = Path(cfg.get("output_dir", "results/raw"))
    output_path = output_dir / f"{result['benchmark_id']}.json"
    save_json(result, output_path)
    print(f"saved: {output_path}")
    print(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LRW BVPSolver geodesic_path benchmark")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
