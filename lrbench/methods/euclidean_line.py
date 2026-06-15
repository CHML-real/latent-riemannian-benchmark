from __future__ import annotations

import torch


def euclidean_line(z0: torch.Tensor, z1: torch.Tensor, num_points: int = 32) -> torch.Tensor:
    """Return a straight latent path from z0 to z1.

    Args:
        z0: Tensor with shape [batch, dim].
        z1: Tensor with shape [batch, dim].
        num_points: Number of path samples, including endpoints.

    Returns:
        Tensor with shape [num_points, batch, dim].
    """
    if z0.shape != z1.shape:
        raise ValueError(f"z0 and z1 must have the same shape, got {z0.shape} and {z1.shape}")
    if z0.ndim != 2:
        raise ValueError(f"expected z0/z1 shape [batch, dim], got {z0.shape}")
    if num_points < 2:
        raise ValueError("num_points must be >= 2")

    t = torch.linspace(0.0, 1.0, num_points, device=z0.device, dtype=z0.dtype)
    t = t.view(num_points, 1, 1)
    return (1.0 - t) * z0.unsqueeze(0) + t * z1.unsqueeze(0)
