from __future__ import annotations

import torch

from lrbench.manifolds.pullback import toy_pullback_metric


def pullback_path_energy(
    path: torch.Tensor,
    beta: float = 6.0,
    sigma: float = 0.7,
    cross: float = 0.2,
    damping: float = 1e-4,
) -> dict[str, float]:
    """Discrete Riemannian path energy under toy pullback metric."""
    if path.ndim != 3 or path.shape[-1] != 2:
        raise ValueError(f"expected path shape [num_points, batch, 2], got {path.shape}")

    num_points = path.shape[0]
    if num_points < 2:
        raise ValueError("path must contain at least two points")

    dz = path[1:] - path[:-1]
    mid = 0.5 * (path[1:] + path[:-1])
    flat_mid = mid.reshape(-1, 2)
    G = toy_pullback_metric(flat_mid, beta=beta, sigma=sigma, cross=cross, damping=damping)
    G = G.reshape(mid.shape[0], mid.shape[1], 2, 2)
    quad = torch.einsum("tbi,tbij,tbj->tb", dz, G, dz)
    dt = 1.0 / float(num_points - 1)
    energy_per_item = quad.sum(dim=0) / dt
    return {
        "pullback_energy_mean": float(energy_per_item.mean().item()),
        "pullback_energy_max": float(energy_per_item.max().item()),
        "pullback_energy_min": float(energy_per_item.min().item()),
    }


def pullback_path_length(
    path: torch.Tensor,
    beta: float = 6.0,
    sigma: float = 0.7,
    cross: float = 0.2,
    damping: float = 1e-4,
) -> dict[str, float]:
    """Discrete Riemannian path length under toy pullback metric."""
    if path.ndim != 3 or path.shape[-1] != 2:
        raise ValueError(f"expected path shape [num_points, batch, 2], got {path.shape}")

    dz = path[1:] - path[:-1]
    mid = 0.5 * (path[1:] + path[:-1])
    flat_mid = mid.reshape(-1, 2)
    G = toy_pullback_metric(flat_mid, beta=beta, sigma=sigma, cross=cross, damping=damping)
    G = G.reshape(mid.shape[0], mid.shape[1], 2, 2)
    quad = torch.einsum("tbi,tbij,tbj->tb", dz, G, dz).clamp_min(0.0)
    length_per_item = torch.sqrt(quad).sum(dim=0)
    return {
        "pullback_length_mean": float(length_per_item.mean().item()),
        "pullback_length_max": float(length_per_item.max().item()),
        "pullback_length_min": float(length_per_item.min().item()),
    }
