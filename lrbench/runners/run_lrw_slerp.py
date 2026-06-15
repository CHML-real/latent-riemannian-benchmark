from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from lrbench.manifolds.sphere import normalize_to_sphere
from lrbench.methods.lrw_slerp import call_lrw_slerp_path, lrw_slerp_metrics, normalize_lrw_slerp_path
from lrbench.methods.sphere_great_circle import sphere_great_circle
from lrbench.utils.device import resolve_device, resolve_dtype
from lrbench.utils.io import load_yaml, save_json


def _make_sphere_pairs(batch_size: int, dimension: int, device: torch.device, dtype: torch.dtype, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    z0 = torch.randn(batch_size, dimension, generator=gen, dtype=dtype).to(device)
    z1 = torch.randn(batch_size, dimension, generator=gen, dtype=dtype).to(device)
    z0 = normalize_to_sphere(z0)
    z1 = normalize_to_sphere(z1)
    # Avoid near-antipodal instability for slerp sanity checks.
    dot = (z0 * z1).sum(dim=-1, keepdim=True)
    z1 = torch.where(dot < -0.85, normalize_to_sphere(z1 + 0.25 * z0), z1)
    return z0, z1


def run(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    device = resolve_device(str(cfg.get("device", "auto")))
    dtype = resolve_dtype(str(cfg.get("dtype", "float32")))
    batch_size = int(cfg.get("batch_size", 1))
    dimension = int(cfg.get("dimension", 3))
    n_points = int(cfg.get("n_points", cfg.get("num_points", 32)))
    use_single_pair = bool(cfg.get("use_single_pair", True))
    endpoint_tolerance = float(cfg.get("endpoint_tolerance", 1e-4))
    sphere_distance_tolerance = float(cfg.get("sphere_distance_tolerance", 1e-4))
    radial_tolerance = float(cfg.get("radial_tolerance", 1e-4))
    reference_error_tolerance = float(cfg.get("reference_error_tolerance", 1e-4))

    z0_batch, z1_batch = _make_sphere_pairs(batch_size, dimension, device, dtype, seed)
    call_z0 = z0_batch[:1] if use_single_pair else z0_batch
    call_z1 = z1_batch[:1] if use_single_pair else z1_batch

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    try:
        raw_path, lrw_info = call_lrw_slerp_path(call_z0, call_z1, n_points=n_points)
        exception_message = None
    except Exception as exc:  # pragma: no cover - exact LRW failures are env-dependent
        raw_path, lrw_info = None, {"available": True, "reason": str(exc)}
        exception_message = str(exc)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    base_result: dict[str, Any] = {
        "benchmark_id": cfg.get("benchmark_id", "lrw_slerp_001"),
        "benchmark_layer": "adapter",
        "target": "lrw_slerp",
        "manifold": "sphere",
        "method": "lrw_slerp_path",
        "seed": seed,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "dimension": dimension,
        "batch_size": 1 if use_single_pair else batch_size,
        "n_points_requested": n_points,
    }

    if raw_path is None:
        reason = exception_message or lrw_info.get("reason") or "LRW slerp unavailable"
        available = bool(lrw_info.get("available", False))
        result = {
            **base_result,
            "status": "skipped" if not available else "failure",
            "lrw": lrw_info,
            "metrics": {
                "available": available,
                "runtime_ms": runtime_ms,
                "peak_memory_mb": peak_memory_mb,
            },
            "failure": {
                "has_failure": bool(available),
                "failure_type": "LRW_F05_SLERP_CALL_FAILED" if available else None,
                "message": reason if available else None,
            },
            "skip": {"is_skipped": not available, "reason": None if available else reason},
        }
    else:
        try:
            path = normalize_lrw_slerp_path(raw_path, call_z0, call_z1, n_points=n_points)
            reference = sphere_great_circle(call_z0, call_z1, num_points=path.shape[0])
            metrics = lrw_slerp_metrics(path, call_z0, call_z1, reference_path=reference)
            metrics["available"] = True
            metrics["runtime_ms"] = runtime_ms
            metrics["peak_memory_mb"] = peak_memory_mb

            has_nan_inf = bool(metrics.get("lrw_slerp_has_nan") or metrics.get("lrw_slerp_has_inf"))
            endpoint_failed = bool(metrics["lrw_slerp_endpoint_error_mean"] > endpoint_tolerance)
            sphere_distance_failed = bool(metrics["lrw_slerp_sphere_distance_error_mean"] > sphere_distance_tolerance)
            radial_failed = bool(metrics["lrw_slerp_sphere_radial_error_max"] > radial_tolerance)
            reference_error = metrics.get("lrw_slerp_vs_reference_relative_error_mean")
            reference_failed = bool(reference_error is not None and reference_error > reference_error_tolerance)
            has_failure = has_nan_inf or endpoint_failed or sphere_distance_failed or radial_failed or reference_failed

            if has_nan_inf:
                failure_type = "GEO_F06_NAN_INF"
                message = "NaN or Inf detected in LRW slerp output"
            elif endpoint_failed:
                failure_type = "GEO_F01_ENDPOINT_MISS"
                message = "LRW slerp path endpoints deviate beyond tolerance"
            elif radial_failed:
                failure_type = "GEO_F14_SPHERE_RADIAL_DRIFT"
                message = "LRW slerp path drifts away from the unit sphere"
            elif sphere_distance_failed:
                failure_type = "GEO_F02_DISTANCE_ERROR"
                message = "LRW slerp path length deviates from sphere ground truth"
            elif reference_failed:
                failure_type = "GEO_F15_REFERENCE_MISMATCH"
                message = "LRW slerp path deviates from reference great-circle implementation"
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
        except Exception as exc:  # pragma: no cover - env-dependent path shapes
            result = {
                **base_result,
                "status": "failure",
                "lrw": lrw_info,
                "metrics": {"available": True, "runtime_ms": runtime_ms, "peak_memory_mb": peak_memory_mb},
                "failure": {"has_failure": True, "failure_type": "LRW_F05_SLERP_EVAL_FAILED", "message": str(exc)},
                "skip": {"is_skipped": False, "reason": None},
            }

    output_dir = Path(cfg.get("output_dir", "results/raw"))
    output_path = output_dir / f"{result['benchmark_id']}.json"
    save_json(result, output_path)
    print(f"saved: {output_path}")
    print(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LRW SLERP sanity benchmark")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
