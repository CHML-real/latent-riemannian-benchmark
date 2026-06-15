from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from lrbench.methods.lrw_geodesic_anatomy import call_lrw_geodesic_outputs, summarize_anatomy
from lrbench.utils.device import resolve_device, resolve_dtype
from lrbench.utils.io import load_yaml, save_json


def _make_endpoint_pairs(batch_size: int, dimension: int, device: torch.device, dtype: torch.dtype, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    z0 = torch.randn(batch_size, dimension, generator=gen, dtype=dtype).to(device)
    z1 = torch.randn(batch_size, dimension, generator=gen, dtype=dtype).to(device)
    return z0, z1


def run(config_path: str) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    device = resolve_device(str(cfg.get("device", "auto")))
    dtype = resolve_dtype(str(cfg.get("dtype", "float32")))
    batch_size = int(cfg.get("batch_size", 1))
    dimension = int(cfg.get("dimension", 2))
    n_points = int(cfg.get("n_points", cfg.get("num_points", 24)))
    endpoint_tolerance = float(cfg.get("endpoint_tolerance", 1e-3))

    z0_batch, z1_batch = _make_endpoint_pairs(batch_size, dimension, device, dtype, seed)
    use_single_pair = bool(cfg.get("use_single_pair", True))
    call_z0 = z0_batch[:1] if use_single_pair else z0_batch
    call_z1 = z1_batch[:1] if use_single_pair else z1_batch

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    anatomy = call_lrw_geodesic_outputs(
        call_z0,
        call_z1,
        n_points=n_points,
        geodesic_n_steps=int(cfg.get("geodesic_n_steps", 100)),
        geodesic_step_size=float(cfg.get("geodesic_step_size", 0.01)),
        bvp_n_steps=int(cfg.get("bvp_n_steps", 20)),
        bvp_step_size=float(cfg.get("bvp_step_size", 0.05)),
        bvp_lr=float(cfg.get("bvp_lr", 0.1)),
        bvp_max_iter=int(cfg.get("bvp_max_iter", 50)),
        bvp_tol=float(cfg.get("bvp_tol", 0.001)),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0

    availability = anatomy.get("availability", {})
    available = bool(availability.get("available", False))
    calls = anatomy.get("calls", [])
    summary = summarize_anatomy(calls, endpoint_tolerance=endpoint_tolerance) if available else {
        "anatomy_call_count": 0,
        "anatomy_successful_call_count": 0,
        "anatomy_axis_candidate_count": 0,
        "anatomy_axis_endpoint_valid_count": 0,
        "anatomy_verdict": "lrw_unavailable",
    }
    metrics = {
        **summary,
        "available": available,
        "runtime_ms": float(runtime_ms),
        "peak_memory_mb": float(peak_memory_mb),
    }

    result: dict[str, Any] = {
        "benchmark_id": cfg.get("benchmark_id", "lrw_geodesic_output_anatomy_001"),
        "benchmark_layer": "adapter",
        "target": "lrw_geodesic_output_anatomy",
        "manifold": "euclidean",
        "method": "lrw_geodesic_output_anatomy",
        "seed": seed,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "dimension": dimension,
        "batch_size": 1 if use_single_pair else batch_size,
        "n_points_requested": n_points,
        "endpoint_tolerance": endpoint_tolerance,
        "status": "skipped" if not available else "success",
        "lrw": availability,
        "endpoints": {"z0": call_z0.detach().cpu().tolist(), "z1": call_z1.detach().cpu().tolist()},
        "summary": summary,
        "metrics": metrics,
        "calls": calls,
        "failure": {"has_failure": False, "failure_type": None, "message": None},
        "skip": {"is_skipped": not available, "reason": availability.get("reason") if not available else None},
    }

    output_dir = Path(cfg.get("output_dir", "results/raw"))
    output_path = output_dir / f"{result['benchmark_id']}.json"
    save_json(result, output_path)
    print(f"saved: {output_path}")
    print(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LRW geodesic solver output anatomy benchmark")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
