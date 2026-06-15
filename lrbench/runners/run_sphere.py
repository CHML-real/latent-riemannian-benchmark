from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from lrbench.manifolds.sphere import normalize_to_sphere, radial_unit_error
from lrbench.methods.sphere_great_circle import sphere_great_circle
from lrbench.metrics.accuracy import endpoint_error, sphere_distance_error
from lrbench.metrics.stability import tensor_stability
from lrbench.utils.device import resolve_device, resolve_dtype
from lrbench.utils.io import load_yaml, save_json


def run(config_path: str) -> dict:
    cfg = load_yaml(config_path)

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    device = resolve_device(str(cfg.get("device", "auto")))
    dtype = resolve_dtype(str(cfg.get("dtype", "float32")))
    ambient_dimension = int(cfg.get("ambient_dimension", cfg.get("dimension", 3)))
    batch_size = int(cfg.get("batch_size", 128))
    num_points = int(cfg.get("num_points", 32))

    if ambient_dimension < 2:
        raise ValueError("Sphere benchmark requires ambient_dimension >= 2")

    z0 = normalize_to_sphere(torch.randn(batch_size, ambient_dimension, device=device, dtype=dtype))
    z1 = normalize_to_sphere(torch.randn(batch_size, ambient_dimension, device=device, dtype=dtype))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    path = sphere_great_circle(z0, z1, num_points=num_points)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_ms = (time.perf_counter() - start) * 1000.0

    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    metrics = {}
    metrics.update(endpoint_error(path, z0, z1))
    metrics.update(sphere_distance_error(path, z0, z1))
    metrics.update(radial_unit_error(path))
    metrics.update(tensor_stability(path))
    metrics["runtime_ms"] = float(runtime_ms)
    metrics["peak_memory_mb"] = float(peak_memory_mb)

    has_failure = bool(metrics["has_nan"] or metrics["has_inf"])
    result = {
        "benchmark_id": cfg.get("benchmark_id", "analytic_sphere_s2_001"),
        "benchmark_layer": "analytic",
        "manifold": cfg.get("manifold", "sphere"),
        "method": cfg.get("method", "sphere_great_circle"),
        "seed": seed,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "ambient_dimension": ambient_dimension,
        "batch_size": batch_size,
        "num_points": num_points,
        "status": "failure" if has_failure else "success",
        "metrics": metrics,
        "failure": {
            "has_failure": has_failure,
            "failure_type": "GEO_F06_NAN_INF" if has_failure else None,
            "message": "NaN or Inf detected in path" if has_failure else None,
        },
    }

    output_dir = Path(cfg.get("output_dir", "results/raw"))
    output_path = output_dir / f"{result['benchmark_id']}.json"
    save_json(result, output_path)
    print(f"saved: {output_path}")
    print(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/analytic_sphere.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
