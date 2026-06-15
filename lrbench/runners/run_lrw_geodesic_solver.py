from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from lrbench.methods.lrw_geodesic_solver import (
    call_lrw_geodesic_solver_euclidean,
    lrw_geodesic_solver_euclidean_metrics,
    normalize_lrw_geodesic_solver_path,
)
from lrbench.utils.device import resolve_device, resolve_dtype
from lrbench.utils.io import load_yaml, save_json


def _make_endpoint_pairs(batch_size: int, dimension: int, device: torch.device, dtype: torch.dtype, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    z0 = torch.randn(batch_size, dimension, generator=gen, dtype=dtype).to(device)
    z1 = torch.randn(batch_size, dimension, generator=gen, dtype=dtype).to(device)
    return z0, z1


def run(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    device = resolve_device(str(cfg.get("device", "auto")))
    dtype = resolve_dtype(str(cfg.get("dtype", "float32")))
    batch_size = int(cfg.get("batch_size", 1))
    dimension = int(cfg.get("dimension", 2))
    n_points = int(cfg.get("n_points", cfg.get("num_points", 24)))
    solver_n_steps = int(cfg.get("solver_n_steps", 100))
    solver_step_size = float(cfg.get("solver_step_size", 0.01))
    endpoint_tolerance = float(cfg.get("endpoint_tolerance", 1e-3))
    distance_tolerance = float(cfg.get("distance_tolerance", 1e-3))
    energy_ratio_tolerance = float(cfg.get("energy_ratio_tolerance", 1.05))
    geodesic_distance_tolerance = float(cfg.get("geodesic_distance_tolerance", 1e-3))

    z0_batch, z1_batch = _make_endpoint_pairs(batch_size, dimension, device, dtype, seed)
    use_single_pair = bool(cfg.get("use_single_pair", True))
    call_z0 = z0_batch[:1] if use_single_pair else z0_batch
    call_z1 = z1_batch[:1] if use_single_pair else z1_batch

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    try:
        raw_path, distance, lrw_info = call_lrw_geodesic_solver_euclidean(
            call_z0,
            call_z1,
            solver_n_steps=solver_n_steps,
            solver_step_size=solver_step_size,
            n_points=n_points,
        )
        exception_message = None
    except Exception as exc:  # pragma: no cover - exact LRW failures are env-dependent
        raw_path, distance, lrw_info = None, None, {"available": True, "reason": str(exc)}
        exception_message = str(exc)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    base_result: dict[str, Any] = {
        "benchmark_id": cfg.get("benchmark_id", "lrw_geodesic_solver_euclidean_001"),
        "benchmark_layer": "adapter",
        "target": "lrw_geodesic_solver",
        "manifold": "euclidean",
        "method": "lrw_geodesic_solver_euclidean",
        "seed": seed,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "dimension": dimension,
        "batch_size": 1 if use_single_pair else batch_size,
        "n_points_requested": n_points,
        "solver_n_steps": solver_n_steps,
        "solver_step_size": solver_step_size,
    }

    if raw_path is None:
        reason = exception_message or lrw_info.get("reason") or "LRW GeodesicSolver unavailable"
        available = bool(lrw_info.get("available", False))
        result = {
            **base_result,
            "status": "skipped" if not available else "failure",
            "lrw": lrw_info,
            "metrics": {
                "available": available,
                "runtime_ms": float(runtime_ms),
                "peak_memory_mb": float(peak_memory_mb),
            },
            "failure": {
                "has_failure": available,
                "failure_type": "LRW_F04_GEODESIC_SOLVER_CALL_FAILED" if available else None,
                "message": reason if available else None,
            },
            "skip": {"is_skipped": not available, "reason": reason if not available else None},
        }
    else:
        path = normalize_lrw_geodesic_solver_path(raw_path, call_z0, call_z1, n_points)
        metrics = lrw_geodesic_solver_euclidean_metrics(path, distance, call_z0, call_z1)
        metrics["available"] = True
        metrics["runtime_ms"] = float(runtime_ms)
        metrics["peak_memory_mb"] = float(peak_memory_mb)

        has_nan_inf = bool(metrics["lrw_geodesic_solver_has_nan"] or metrics["lrw_geodesic_solver_has_inf"])
        distance_nan_inf = bool(metrics.get("lrw_geodesic_solver_distance_has_nan") or metrics.get("lrw_geodesic_solver_distance_has_inf"))
        endpoint_failed = bool(metrics["lrw_geodesic_solver_endpoint_error_mean"] > endpoint_tolerance)
        path_distance_failed = bool(metrics["lrw_geodesic_solver_distance_error_mean"] > distance_tolerance)
        energy_failed = bool(metrics["lrw_geodesic_solver_energy_ratio_over_straight"] > energy_ratio_tolerance)
        geodesic_distance_error = metrics.get("lrw_geodesic_solver_geodesic_distance_error_mean")
        geodesic_distance_failed = bool(geodesic_distance_error is not None and geodesic_distance_error > geodesic_distance_tolerance)
        has_failure = has_nan_inf or distance_nan_inf or endpoint_failed or path_distance_failed or energy_failed or geodesic_distance_failed

        if has_nan_inf or distance_nan_inf:
            failure_type = "GEO_F06_NAN_INF"
            message = "NaN or Inf detected in LRW GeodesicSolver output"
        elif endpoint_failed:
            failure_type = "GEO_F01_ENDPOINT_MISS"
            message = "LRW GeodesicSolver interpolate path endpoints deviate beyond tolerance"
        elif path_distance_failed:
            failure_type = "GEO_F02_DISTANCE_ERROR"
            message = "LRW GeodesicSolver interpolate path length deviates from Euclidean ground truth"
        elif geodesic_distance_failed:
            failure_type = "GEO_F02_GEODESIC_DISTANCE_ERROR"
            message = "LRW GeodesicSolver geodesic_distance deviates from Euclidean ground truth"
        elif energy_failed:
            failure_type = "GEO_F03_ENERGY_NOT_OPTIMAL"
            message = "LRW GeodesicSolver path energy is worse than straight-line tolerance"
        else:
            failure_type = None
            message = None

        result = {
            **base_result,
            "status": "failure" if has_failure else "success",
            "lrw": lrw_info,
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
    parser = argparse.ArgumentParser(description="Run LRW GeodesicSolver Euclidean sanity benchmark")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
