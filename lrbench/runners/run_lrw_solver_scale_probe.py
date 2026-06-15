from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from lrbench.methods.lrw_solver_scale_probe import run_lrw_solver_scale_probe
from lrbench.utils.device import resolve_device, resolve_dtype
from lrbench.utils.io import load_yaml, save_json


def _make_endpoint_pairs(batch_size: int, dimension: int, device: torch.device, dtype: torch.dtype, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    z0 = torch.randn(batch_size, dimension, generator=gen, dtype=dtype).to(device)
    z1 = torch.randn(batch_size, dimension, generator=gen, dtype=dtype).to(device)
    return z0, z1


def _as_int_list(value: Any, default: list[int]) -> list[int]:
    if value is None:
        return default
    if isinstance(value, list):
        return [int(v) for v in value]
    return [int(value)]


def _as_float_list(value: Any, default: list[float]) -> list[float]:
    if value is None:
        return default
    if isinstance(value, list):
        return [float(v) for v in value]
    return [float(value)]


def run(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    device = resolve_device(str(cfg.get("device", "auto")))
    dtype = resolve_dtype(str(cfg.get("dtype", "float32")))
    batch_size = int(cfg.get("batch_size", 1))
    dimension = int(cfg.get("dimension", 2))
    n_points = int(cfg.get("n_points", cfg.get("num_points", 24)))
    n_steps_values = _as_int_list(cfg.get("n_steps_values"), [1, 10, 50, 100])
    step_size_values = _as_float_list(cfg.get("step_size_values"), [0.001, 0.01, 0.05, 0.1, 1.0])

    z0_batch, z1_batch = _make_endpoint_pairs(batch_size, dimension, device, dtype, seed)
    use_single_pair = bool(cfg.get("use_single_pair", True))
    call_z0 = z0_batch[:1] if use_single_pair else z0_batch
    call_z1 = z1_batch[:1] if use_single_pair else z1_batch

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    rows, lrw_info, summary = run_lrw_solver_scale_probe(
        call_z0,
        call_z1,
        n_points=n_points,
        n_steps_values=n_steps_values,
        step_size_values=step_size_values,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    metrics: dict[str, Any] = {**summary, "available": bool(lrw_info.get("available", False)), "runtime_ms": runtime_ms, "peak_memory_mb": peak_memory_mb}
    status = "success"
    skip = {"is_skipped": False, "reason": None}
    failure = {"has_failure": False, "failure_type": None, "message": None}
    if not lrw_info.get("available", False):
        status = "skipped"
        skip = {"is_skipped": True, "reason": lrw_info.get("reason", "LRW GeodesicSolver unavailable")}
    elif summary.get("scale_probe_success_count", 0) == 0:
        status = "failure"
        failure = {"has_failure": True, "failure_type": "GEO_F08_SCALE_PROBE_CALLS_FAILED", "message": "All LRW solver scale probe calls failed"}

    result = {
        "benchmark_id": cfg.get("benchmark_id", "lrw_solver_scale_probe_001"),
        "benchmark_layer": "adapter",
        "target": "lrw_solver_scale_probe",
        "manifold": "euclidean",
        "method": "lrw_solver_scale_probe",
        "seed": seed,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "dimension": dimension,
        "batch_size": 1 if use_single_pair else batch_size,
        "n_points_requested": n_points,
        "n_steps_values": n_steps_values,
        "step_size_values": step_size_values,
        "endpoints": {"z0": call_z0.detach().cpu().tolist(), "z1": call_z1.detach().cpu().tolist()},
        "status": status,
        "lrw": lrw_info,
        "summary": summary,
        "metrics": metrics,
        "cases": rows,
        "failure": failure,
        "skip": skip,
    }

    output_dir = Path(cfg.get("output_dir", "results/raw"))
    output_path = output_dir / f"{result['benchmark_id']}.json"
    save_json(result, output_path)
    print(f"saved: {output_path}")
    print(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LRW solver step-scale probe")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
