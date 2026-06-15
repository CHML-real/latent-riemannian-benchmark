from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from lrbench.methods.euclidean_line import euclidean_line
from lrbench.metrics.accuracy import endpoint_error, euclidean_distance_error
from lrbench.metrics.scaling import group_by_batch_size, group_by_dimension, summarize_stress_cases
from lrbench.metrics.stability import tensor_stability
from lrbench.utils.device import resolve_device, resolve_dtype
from lrbench.utils.io import load_yaml, save_json


def _as_list(value, default):
    if value is None:
        return default
    if isinstance(value, list):
        return value
    return [value]


def _make_random_pairs(batch_size: int, dimension: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    z0 = torch.randn(batch_size, dimension, device=device, dtype=dtype)
    z1 = torch.randn(batch_size, dimension, device=device, dtype=dtype)
    return z0, z1


def _run_case(
    benchmark_id: str,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    dimension: int,
    batch_size: int,
    num_points: int,
    max_endpoint_error: float,
    max_distance_error: float,
) -> dict:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    try:
        z0, z1 = _make_random_pairs(batch_size, dimension, device, dtype)
        start = time.perf_counter()
        path = euclidean_line(z0, z1, num_points=num_points)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        runtime_ms = (time.perf_counter() - start) * 1000.0

        peak_memory_mb = 0.0
        if device.type == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        metrics = {}
        metrics.update(endpoint_error(path, z0, z1))
        metrics.update(euclidean_distance_error(path, z0, z1))
        metrics.update(tensor_stability(path))
        metrics["runtime_ms"] = float(runtime_ms)
        metrics["peak_memory_mb"] = float(peak_memory_mb)
        metrics["path_numel"] = int(path.numel())

        has_nan_inf = bool(metrics["has_nan"] or metrics["has_inf"])
        endpoint_failed = bool(metrics["endpoint_error_mean"] > max_endpoint_error)
        distance_failed = bool(metrics["distance_error_mean"] > max_distance_error)
        has_failure = has_nan_inf or endpoint_failed or distance_failed

        if has_nan_inf:
            failure_type = "GEO_F06_NAN_INF"
            message = "NaN or Inf detected in stress benchmark path"
        elif endpoint_failed:
            failure_type = "GEO_F01_ENDPOINT_MISS"
            message = "endpoint error exceeded configured threshold"
        elif distance_failed:
            failure_type = "GEO_F02_DISTANCE_EXPLOSION"
            message = "distance error exceeded configured threshold"
        else:
            failure_type = None
            message = None

        status = "failure" if has_failure else "success"
    except Exception as exc:  # deliberately convert stress failures into result rows
        metrics = {
            "runtime_ms": 0.0,
            "peak_memory_mb": 0.0,
            "path_numel": 0,
            "has_nan": True,
            "has_inf": False,
            "nan_rate": 1.0,
            "inf_rate": 0.0,
            "endpoint_error_mean": float("inf"),
            "distance_error_mean": float("inf"),
        }
        status = "failure"
        failure_type = "GEO_F14_RUNTIME_EXCEPTION"
        message = repr(exc)

    return {
        "case_id": f"{benchmark_id}_dim{dimension}_batch{batch_size}_points{num_points}",
        "benchmark_layer": "stress",
        "manifold": "euclidean",
        "method": "euclidean_line_scaling",
        "seed": seed,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "dimension": dimension,
        "batch_size": batch_size,
        "num_points": num_points,
        "status": status,
        "metrics": metrics,
        "failure": {
            "has_failure": status == "failure",
            "failure_type": failure_type,
            "message": message,
        },
    }


def run(config_path: str) -> dict:
    cfg = load_yaml(config_path)

    benchmark_id = str(cfg.get("benchmark_id", "stress_scaling_001"))
    base_seed = int(cfg.get("seed", 42))
    device = resolve_device(str(cfg.get("device", "auto")))
    dtype = resolve_dtype(str(cfg.get("dtype", "float32")))
    dimensions = [int(x) for x in _as_list(cfg.get("dimensions"), [2, 8, 32, 128])]
    batch_sizes = [int(x) for x in _as_list(cfg.get("batch_sizes"), [1, 16, 64])]
    num_points_list = [int(x) for x in _as_list(cfg.get("num_points"), [32, 64])]
    max_endpoint_error = float(cfg.get("max_endpoint_error", 1e-5))
    max_distance_error = float(cfg.get("max_distance_error", 1e-5))

    cases = []
    case_index = 0
    total_start = time.perf_counter()
    for dimension in dimensions:
        for batch_size in batch_sizes:
            for num_points in num_points_list:
                case_seed = base_seed + case_index
                case = _run_case(
                    benchmark_id=benchmark_id,
                    seed=case_seed,
                    device=device,
                    dtype=dtype,
                    dimension=dimension,
                    batch_size=batch_size,
                    num_points=num_points,
                    max_endpoint_error=max_endpoint_error,
                    max_distance_error=max_distance_error,
                )
                cases.append(case)
                case_index += 1
    total_runtime_ms = (time.perf_counter() - total_start) * 1000.0

    summary = summarize_stress_cases(cases)
    summary["total_runtime_ms"] = float(total_runtime_ms)

    result = {
        "benchmark_id": benchmark_id,
        "benchmark_layer": "stress",
        "target": "euclidean_line_scaling",
        "seed": base_seed,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "dimensions": dimensions,
        "batch_sizes": batch_sizes,
        "num_points": num_points_list,
        "status": "failure" if summary["failure_count"] > 0 else "success",
        "summary": summary,
        "by_dimension": group_by_dimension(cases),
        "by_batch_size": group_by_batch_size(cases),
        "cases": cases,
        "failure": {
            "has_failure": summary["failure_count"] > 0,
            "failure_type": "GEO_F15_STRESS_CASE_FAILURE" if summary["failure_count"] > 0 else None,
            "message": f"{summary['failure_count']} stress cases failed" if summary["failure_count"] > 0 else None,
        },
    }

    output_dir = Path(cfg.get("output_dir", "results/raw"))
    output_path = output_dir / f"{benchmark_id}.json"
    save_json(result, output_path)
    print(f"saved: {output_path}")
    print({
        "benchmark_id": result["benchmark_id"],
        "status": result["status"],
        "summary": result["summary"],
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stress_scaling.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
