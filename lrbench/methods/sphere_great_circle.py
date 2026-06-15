from __future__ import annotations

import torch

from lrbench.manifolds.sphere import normalize_to_sphere


def sphere_great_circle(z0: torch.Tensor, z1: torch.Tensor, num_points: int = 32, eps: float = 1e-7) -> torch.Tensor:
    """Spherical linear interpolation along the great circle on S^{d-1}.

    Args:
        z0: Unit start vectors, shape (batch, dim).
        z1: Unit end vectors, shape (batch, dim).
        num_points: Number of path points including endpoints.
        eps: Numerical stabilizer for small angles.

    Returns:
        Tensor with shape (num_points, batch, dim).
    """
    z0 = normalize_to_sphere(z0)
    z1 = normalize_to_sphere(z1)

    dot = (z0 * z1).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.arccos(dot)
    sin_omega = torch.sin(omega).clamp_min(eps)

    t = torch.linspace(0.0, 1.0, num_points, device=z0.device, dtype=z0.dtype).view(num_points, 1, 1)
    z0_b = z0.unsqueeze(0)
    z1_b = z1.unsqueeze(0)
    omega_b = omega.unsqueeze(0)
    sin_omega_b = sin_omega.unsqueeze(0)

    path = (torch.sin((1.0 - t) * omega_b) / sin_omega_b) * z0_b + (torch.sin(t * omega_b) / sin_omega_b) * z1_b
    return normalize_to_sphere(path)
