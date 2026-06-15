from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from lrbench.decoders.toy_decoder import gaussian_bump_decoder
from lrbench.methods.euclidean_line import euclidean_line
from lrbench.methods.pullback_paths import pullback_bump_detour_path
from lrbench.methods.lrw_pullback_metric import (
    compare_metric_tensors,
    compute_lrw_pullback_metric,
    compute_reference_pullback_metric,
    metric_tensor_stats,
)
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


def _sample_points_for_metric(
    cfg: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    batch_size = int(cfg.get("batch_size", 32))
    num_points = int(cfg.get("num_points", 48))
    left_x = float(cfg.get("left_x", -2.0))
    right_x = float(cfg.get("right_x", 2.0))
    base_y = float(cfg.get("base_y", 0.0))
    detour_y = float(cfg.get("detour_y", 1.5))
    max_metric_points = int(cfg.get("max_metric_points", 256))

    z0, z1 = _make_endpoint_pairs(batch_size, left_x, right_x, base_y, device, dtype)
    straight = euclidean_line(z0, z1, num_points=num_points)
    detour = pullback_bump_detour_path(z0, z1, detour_y=detour_y, num_points=num_points)
    points = torch.cat([straight, detour], dim=0).reshape(-1, 2)
    if points.shape[0] > max_metric_points:
        idx = torch.linspace(0, points.shape[0] - 1, max_metric_points, device=device).long()
        points = points.index_select(0, idx)
    return points.contiguous()


def run(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    device = resolve_device(str(cfg.get("device", "auto")))
    dtype = resolve_dtype(str(cfg.get("dtype", "float32")))
    beta = float(cfg.get("beta", 6.0))
    sigma = float(cfg.get("sigma", 0.7))
    cross = float(cfg.get("cross", 0.2))
    regularization = float(cfg.get("regularization", cfg.get("damping", 1e-5)))
    chunk_size_value = cfg.get("chunk_size", None)
    chunk_size = None if chunk_size_value in (None, "null", "None") else int(chunk_size_value)

    z = _sample_points_for_metric(cfg, device=device, dtype=dtype)

    def decoder(x: torch.Tensor) -> torch.Tensor:
        return gaussian_bump_decoder(x, beta=beta, sigma=sigma, cross=cross)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    lrw_G, lrw_info = compute_lrw_pullback_metric(
        z,
        decoder=decoder,
        regularization=regularization,
        chunk_size=chunk_size,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_ms = (time.perf_counter() - start) * 1000.0

    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    if lrw_G is None:
        result = {
            "benchmark_id": cfg.get("benchmark_id", "lrw_pullback_metric_001"),
            "benchmark_layer": "adapter",
            "target": "lrw_pullback_metric",
            "manifold": "toy_decoder_pullback_metric",
            "method": "lrw_pullback_metric_tensor",
            "status": "skipped",
            "seed": seed,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "latent_dimension": 2,
            "sample_count": int(z.shape[0]),
            "regularization": regularization,
            "chunk_size": chunk_size,
            "metrics": {
                "available": False,
                "runtime_ms": float(runtime_ms),
                "peak_memory_mb": float(peak_memory_mb),
            },
            "failure": {"has_failure": False, "failure_type": None, "message": None},
            "skip": {"is_skipped": True, "reason": lrw_info.get("reason") or "LRW PullbackMetric unavailable"},
        }
    else:
        ref_G = compute_reference_pullback_metric(
            z,
            beta=beta,
            sigma=sigma,
            cross=cross,
            damping=regularization,
        )
        metrics = {}
        metrics.update(metric_tensor_stats(lrw_G, prefix="lrw_pullback"))
        metrics.update(metric_tensor_stats(ref_G, prefix="reference_pullback"))
        metrics.update(compare_metric_tensors(lrw_G, ref_G))
        metrics["available"] = True
        metrics["sample_count"] = int(z.shape[0])
        metrics["runtime_ms"] = float(runtime_ms)
        metrics["peak_memory_mb"] = float(peak_memory_mb)

        has_nan_inf = bool(metrics["lrw_pullback_has_nan"] or metrics["lrw_pullback_has_inf"])
        spd_failed = bool(metrics["lrw_pullback_spd_violation_rate"] > 0.0)
        shape_failed = not bool(metrics["lrw_vs_reference_same_shape"])
        tolerance = float(cfg.get("relative_error_tolerance", 1e-4))
        rel_error_failed = bool(metrics["lrw_vs_reference_relative_frobenius_error_mean"] > tolerance)
        compare_reference = bool(cfg.get("compare_reference", True))
        has_failure = has_nan_inf or spd_failed or shape_failed or (compare_reference and rel_error_failed)

        if has_nan_inf:
            failure_type = "GEO_F06_NAN_INF"
            message = "NaN or Inf detected in LRW pullback metric tensor"
        elif spd_failed:
            failure_type = "GEO_F04_NON_SPD_METRIC"
            message = "LRW PullbackMetric has non-positive eigenvalues"
        elif shape_failed:
            failure_type = "GEO_F10_BATCH_SHAPE_MISMATCH"
            message = "LRW PullbackMetric tensor shape differs from reference"
        elif compare_reference and rel_error_failed:
            failure_type = "LRW_F01_PULLBACK_REFERENCE_MISMATCH"
            message = "LRW PullbackMetric differs from lrbench reference above tolerance"
        else:
            failure_type = None
            message = None

        result = {
            "benchmark_id": cfg.get("benchmark_id", "lrw_pullback_metric_001"),
            "benchmark_layer": "adapter",
            "target": "lrw_pullback_metric",
            "manifold": "toy_decoder_pullback_metric",
            "method": "lrw_pullback_metric_tensor",
            "status": "failure" if has_failure else "success",
            "seed": seed,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "latent_dimension": 2,
            "decoder_output_dimension": 4,
            "sample_count": int(z.shape[0]),
            "regularization": regularization,
            "chunk_size": chunk_size,
            "beta": beta,
            "sigma": sigma,
            "cross": cross,
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
    parser = argparse.ArgumentParser(description="Run LRW PullbackMetric tensor benchmark")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
