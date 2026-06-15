from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from lrbench.manifolds.pullback import pullback_metric_eigen_stats
from lrbench.methods.euclidean_line import euclidean_line
from lrbench.methods.pullback_paths import pullback_bump_detour_path
from lrbench.metrics.accuracy import endpoint_error
from lrbench.metrics.pullback_energy import pullback_path_energy, pullback_path_length
from lrbench.metrics.stability import tensor_stability
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


def run(config_path: str) -> dict:
    cfg = load_yaml(config_path)

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    device = resolve_device(str(cfg.get("device", "auto")))
    dtype = resolve_dtype(str(cfg.get("dtype", "float32")))
    batch_size = int(cfg.get("batch_size", 128))
    num_points = int(cfg.get("num_points", 128))
    left_x = float(cfg.get("left_x", -2.0))
    right_x = float(cfg.get("right_x", 2.0))
    base_y = float(cfg.get("base_y", 0.0))
    detour_y = float(cfg.get("detour_y", 1.5))
    beta = float(cfg.get("beta", 6.0))
    sigma = float(cfg.get("sigma", 0.7))
    cross = float(cfg.get("cross", 0.2))
    damping = float(cfg.get("damping", 1e-4))

    z0, z1 = _make_endpoint_pairs(batch_size, left_x, right_x, base_y, device, dtype)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    straight_path = euclidean_line(z0, z1, num_points=num_points)
    detour_path = pullback_bump_detour_path(z0, z1, detour_y=detour_y, num_points=num_points)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_ms = (time.perf_counter() - start) * 1000.0

    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    straight_energy = pullback_path_energy(straight_path, beta=beta, sigma=sigma, cross=cross, damping=damping)
    detour_energy = pullback_path_energy(detour_path, beta=beta, sigma=sigma, cross=cross, damping=damping)
    straight_length = pullback_path_length(straight_path, beta=beta, sigma=sigma, cross=cross, damping=damping)
    detour_length = pullback_path_length(detour_path, beta=beta, sigma=sigma, cross=cross, damping=damping)

    straight_energy_mean = straight_energy["pullback_energy_mean"]
    detour_energy_mean = detour_energy["pullback_energy_mean"]
    straight_length_mean = straight_length["pullback_length_mean"]
    detour_length_mean = detour_length["pullback_length_mean"]

    energy_ratio = detour_energy_mean / max(straight_energy_mean, 1e-12)
    length_ratio = detour_length_mean / max(straight_length_mean, 1e-12)
    energy_improvement = 1.0 - energy_ratio
    length_improvement = 1.0 - length_ratio

    sample_points = torch.cat([straight_path, detour_path], dim=0).reshape(-1, 2)

    metrics = {}
    metrics.update({f"straight_{k}": v for k, v in endpoint_error(straight_path, z0, z1).items()})
    metrics.update({f"detour_{k}": v for k, v in endpoint_error(detour_path, z0, z1).items()})
    metrics.update({f"straight_{k}": v for k, v in straight_energy.items()})
    metrics.update({f"detour_{k}": v for k, v in detour_energy.items()})
    metrics.update({f"straight_{k}": v for k, v in straight_length.items()})
    metrics.update({f"detour_{k}": v for k, v in detour_length.items()})
    metrics.update(pullback_metric_eigen_stats(sample_points, beta=beta, sigma=sigma, cross=cross, damping=damping))
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
    spd_failed = bool(metrics["pullback_spd_violation_rate"] > 0.0)
    expected_improvement = bool(cfg.get("expected_detour_better", True))
    failed_improvement = expected_improvement and not (energy_ratio < 1.0 and length_ratio < 1.0)
    has_failure = has_nan_inf or spd_failed or failed_improvement

    if has_nan_inf:
        failure_type = "GEO_F06_NAN_INF"
        message = "NaN or Inf detected in pullback benchmark paths"
    elif spd_failed:
        failure_type = "GEO_F04_NON_SPD_METRIC"
        message = "pullback metric has non-positive eigenvalues"
    elif failed_improvement:
        failure_type = "GEO_F03_ENERGY_NOT_DECREASING"
        message = "detour path did not reduce energy/length under toy pullback metric"
    else:
        failure_type = None
        message = None

    result = {
        "benchmark_id": cfg.get("benchmark_id", "synthetic_pullback_001"),
        "benchmark_layer": "synthetic",
        "manifold": "toy_decoder_pullback_metric",
        "method": "straight_vs_pullback_detour",
        "seed": seed,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "latent_dimension": 2,
        "decoder_output_dimension": 4,
        "batch_size": batch_size,
        "num_points": num_points,
        "left_x": left_x,
        "right_x": right_x,
        "base_y": base_y,
        "detour_y": detour_y,
        "beta": beta,
        "sigma": sigma,
        "cross": cross,
        "damping": damping,
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
    parser.add_argument("--config", type=str, default="configs/synthetic_pullback.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
