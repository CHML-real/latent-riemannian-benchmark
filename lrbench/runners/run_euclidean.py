from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from lrbench.methods.euclidean_line import euclidean_line
from lrbench.metrics.accuracy import endpoint_error, euclidean_distance_error
from lrbench.metrics.geometry import euclidean_path_energy
from lrbench.metrics.stability import tensor_stability
from lrbench.utils.device import resolve_device, resolve_dtype
from lrbench.utils.io import load_yaml, save_json


def run(config_path: str) -> dict:
    cfg = load_yaml(config_path)

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    device = resolve_device(str(cfg.get("device", "auto")))
    dtype = resolve_dtype(str(cfg.get("dtype", "float32")))
    dimension = int(cfg.get("dimension", 8))
    batch_size = int(cfg.get("batch_size", 128))
    num_points = int(cfg.get("num_points", 32))

    z0 = torch.randn(batch_size, dimension, device=device, dtype=dtype)
    z1 = torch.randn(batch_size, dimension, device=device, dtype=dtype)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

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
    metrics.update(euclidean_path_energy(path))
    metrics.update(tensor_stability(path))
    metrics["runtime_ms"] = float(runtime_ms)
    metrics["peak_memory_mb"] = float(peak_memory_mb)

    has_failure = bool(metrics["has_nan"] or metrics["has_inf"])
    result = {
        "benchmark_id": cfg.get("benchmark_id", "analytic_euclidean_001"),
        "benchmark_layer": "analytic",
        "manifold": "euclidean",
        "method": cfg.get("method", "euclidean_line"),
        "seed": seed,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "dimension": dimension,
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
    parser.add_argument("--config", type=str, default="configs/euclidean_default.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
