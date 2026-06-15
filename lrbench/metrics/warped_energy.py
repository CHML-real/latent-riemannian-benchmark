from __future__ import annotations

import torch

from lrbench.manifolds.warped import diagonal_warped_metric


def warped_path_energy(path: torch.Tensor, alpha: float = 2.0) -> dict[str, float]:
    """Discrete Riemannian path energy under the diagonal warped metric.

    Energy is approximated with midpoint metric evaluation:

        E ~= sum_i (delta_i^T G(mid_i) delta_i) / dt

    where dt = 1 / (num_points - 1). The absolute value depends on sampling,
    but comparisons between paths with the same num_points are meaningful.
    """
    if path.ndim != 3 or path.shape[-1] != 2:
        raise ValueError(f"expected path shape [num_points, batch, 2], got {path.shape}")

    num_points = path.shape[0]
    if num_points < 2:
        raise ValueError("path must contain at least two points")

    dz = path[1:] - path[:-1]
    mid = 0.5 * (path[1:] + path[:-1])
    G = diagonal_warped_metric(mid, alpha=alpha)
    quad = torch.einsum("tbi,tbij,tbj->tb", dz, G, dz)
    dt = 1.0 / float(num_points - 1)
    energy_per_item = quad.sum(dim=0) / dt
    return {
        "warped_energy_mean": float(energy_per_item.mean().item()),
        "warped_energy_max": float(energy_per_item.max().item()),
        "warped_energy_min": float(energy_per_item.min().item()),
    }


def warped_path_length(path: torch.Tensor, alpha: float = 2.0) -> dict[str, float]:
    """Discrete Riemannian path length under the diagonal warped metric."""
    if path.ndim != 3 or path.shape[-1] != 2:
        raise ValueError(f"expected path shape [num_points, batch, 2], got {path.shape}")

    dz = path[1:] - path[:-1]
    mid = 0.5 * (path[1:] + path[:-1])
    G = diagonal_warped_metric(mid, alpha=alpha)
    quad = torch.einsum("tbi,tbij,tbj->tb", dz, G, dz).clamp_min(0.0)
    length_per_item = torch.sqrt(quad).sum(dim=0)
    return {
        "warped_length_mean": float(length_per_item.mean().item()),
        "warped_length_max": float(length_per_item.max().item()),
        "warped_length_min": float(length_per_item.min().item()),
    }
