from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from lrbench.manifolds.warped import metric_eigen_stats
from lrbench.methods.euclidean_line import euclidean_line
from lrbench.methods.warped_paths import warped_detour_path
from lrbench.metrics.accuracy import endpoint_error
from lrbench.metrics.stability import tensor_stability
from lrbench.metrics.warped_energy import warped_path_energy, warped_path_length
from lrbench.utils.device import resolve_device, resolve_dtype
from lrbench.utils.io import load_yaml, save_json


def _make_endpoint_pairs(
    batch_size: int,
    left_x: float,
    y_span: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic endpoint pairs that expose the warped metric effect."""
    z0 = torch.empty(batch_size, 2, device=device, dtype=dtype)
    z1 = torch.empty(batch_size, 2, device=device, dtype=dtype)
    z0[:, 0] = left_x
    z1[:, 0] = left_x
    z0[:, 1] = -0.5 * y_span
    z1[:, 1] = 0.5 * y_span
    return z0, z1


def run(config_path: str) -> dict:
    cfg = load_yaml(config_path)

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    device = resolve_device(str(cfg.get("device", "auto")))
    dtype = resolve_dtype(str(cfg.get("dtype", "float32")))
    batch_size = int(cfg.get("batch_size", 128))
    num_points = int(cfg.get("num_points", 96))
    alpha = float(cfg.get("alpha", 2.0))
    left_x = float(cfg.get("left_x", -1.0))
    detour_x = float(cfg.get("detour_x", 1.0))
    y_span = float(cfg.get("y_span", 8.0))

    z0, z1 = _make_endpoint_pairs(batch_size, left_x, y_span, device, dtype)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    straight_path = euclidean_line(z0, z1, num_points=num_points)
    detour_path = warped_detour_path(z0, z1, detour_x=detour_x, num_points=num_points)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_ms = (time.perf_counter() - start) * 1000.0

    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    straight_energy = warped_path_energy(straight_path, alpha=alpha)
    detour_energy = warped_path_energy(detour_path, alpha=alpha)
    straight_length = warped_path_length(straight_path, alpha=alpha)
    detour_length = warped_path_length(detour_path, alpha=alpha)

    straight_energy_mean = straight_energy["warped_energy_mean"]
    detour_energy_mean = detour_energy["warped_energy_mean"]
    straight_length_mean = straight_length["warped_length_mean"]
    detour_length_mean = detour_length["warped_length_mean"]

    energy_ratio = detour_energy_mean / max(straight_energy_mean, 1e-12)
    length_ratio = detour_length_mean / max(straight_length_mean, 1e-12)
    energy_improvement = 1.0 - energy_ratio
    length_improvement = 1.0 - length_ratio

    metrics = {}
    metrics.update({f"straight_{k}": v for k, v in endpoint_error(straight_path, z0, z1).items()})
    metrics.update({f"detour_{k}": v for k, v in endpoint_error(detour_path, z0, z1).items()})
    metrics.update({f"straight_{k}": v for k, v in straight_energy.items()})
    metrics.update({f"detour_{k}": v for k, v in detour_energy.items()})
    metrics.update({f"straight_{k}": v for k, v in straight_length.items()})
    metrics.update({f"detour_{k}": v for k, v in detour_length.items()})
    metrics.update(metric_eigen_stats(torch.cat([straight_path, detour_path], dim=0).reshape(-1, 2), alpha=alpha))
    metrics.update({f"straight_{k}": v for k, v in tensor_stability(straight_path).items()})
    metrics.update({f"detour_{k}": v for k, v in tensor_stability(detour_path).items()})
    metrics["energy_ratio_detour_over_straight"] = float(energy_ratio)
    metrics["length_ratio_detour_over_straight"] = float(length_ratio)
    metrics["energy_improvement"] = float(energy_improvement)
    metrics["length_improvement"] = float(length_improvement)
    metrics["runtime_ms"] = float(runtime_ms)
    metrics["peak_memory_mb"] = float(peak_memory_mb)

    has_nan_inf = bool(
        metrics["straight_has_nan"]
        or metrics["straight_has_inf"]
        or metrics["detour_has_nan"]
        or metrics["detour_has_inf"]
    )
    expected_improvement = bool(cfg.get("expected_detour_better", True))
    failed_improvement = expected_improvement and not (energy_ratio < 1.0 and length_ratio < 1.0)
    has_failure = has_nan_inf or failed_improvement

    if has_nan_inf:
        failure_type = "GEO_F06_NAN_INF"
        message = "NaN or Inf detected in warped benchmark paths"
    elif failed_improvement:
        failure_type = "GEO_F03_ENERGY_NOT_DECREASING"
        message = "detour path did not reduce energy/length under warped metric"
    else:
        failure_type = None
        message = None

    result = {
        "benchmark_id": cfg.get("benchmark_id", "synthetic_warped_001"),
        "benchmark_layer": "synthetic",
        "manifold": cfg.get("manifold", "diagonal_warped_metric"),
        "method": cfg.get("method", "straight_vs_warped_detour"),
        "seed": seed,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "dimension": 2,
        "batch_size": batch_size,
        "num_points": num_points,
        "alpha": alpha,
        "left_x": left_x,
        "detour_x": detour_x,
        "y_span": y_span,
        "status": "failure" if has_failure else "success",
        "metrics": metrics,
        "failure": {
            "has_failure": has_failure,
            "failure_type": failure_type,
            "message": message,
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
    parser.add_argument("--config", type=str, default="configs/synthetic_warped.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
