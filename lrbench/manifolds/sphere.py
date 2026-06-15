from __future__ import annotations

import torch


def normalize_to_sphere(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Project nonzero vectors onto the unit sphere S^{d-1}."""
    norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(eps)
    return x / norm


def sphere_geodesic_distance(z0: torch.Tensor, z1: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Great-circle distance on the unit sphere.

    z0 and z1 are expected to be unit vectors with shape (..., dim).
    The returned distance has shape (...,).
    """
    dot = (z0 * z1).sum(dim=-1).clamp(-1.0 + eps, 1.0)
    return torch.arccos(dot)


def sphere_path_length(path: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Discrete great-circle length of a path on the unit sphere.

    Uses the chord formula ``2 * asin(||a-b|| / 2)`` instead of
    ``acos(dot(a, b))`` because the chord formula is more stable for
    very small adjacent steps in float32.

    path shape: (num_points, batch, dim)
    returns: (batch,)
    """
    a = path[:-1].to(torch.float64)
    b = path[1:].to(torch.float64)
    chord = torch.linalg.norm(a - b, dim=-1).clamp(0.0, 2.0 - eps)
    return (2.0 * torch.arcsin(0.5 * chord)).sum(dim=0).to(path.dtype)


def radial_unit_error(path: torch.Tensor) -> dict[str, float]:
    """Measure how much a path drifts away from the unit sphere."""
    radii = torch.linalg.norm(path, dim=-1)
    err = torch.abs(radii - 1.0)
    return {
        "sphere_radial_error_mean": float(err.mean().item()),
        "sphere_radial_error_max": float(err.max().item()),
    }
