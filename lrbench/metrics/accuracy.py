from __future__ import annotations

import torch


def endpoint_error(path: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor) -> dict[str, float]:
    """Measure how exactly the path starts/ends at the requested endpoints."""
    start_error = torch.linalg.norm(path[0] - z0, dim=-1)
    end_error = torch.linalg.norm(path[-1] - z1, dim=-1)
    mean_value = 0.5 * (start_error.mean() + end_error.mean())
    return {
        "endpoint_error_start_mean": float(start_error.mean().item()),
        "endpoint_error_end_mean": float(end_error.mean().item()),
        "endpoint_error_mean": float(mean_value.item()),
        "endpoint_error_max": float(torch.maximum(start_error, end_error).max().item()),
    }


def euclidean_distance_error(path: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor, eps: float = 1e-8) -> dict[str, float]:
    """Compare discrete path length with true Euclidean endpoint distance."""
    true_distance = torch.linalg.norm(z1 - z0, dim=-1)
    diffs = path[1:] - path[:-1]
    path_distance = torch.linalg.norm(diffs, dim=-1).sum(dim=0)
    rel_error = torch.abs(path_distance - true_distance) / torch.clamp(true_distance, min=eps)
    return {
        "true_distance_mean": float(true_distance.mean().item()),
        "path_distance_mean": float(path_distance.mean().item()),
        "distance_error_mean": float(rel_error.mean().item()),
        "distance_error_max": float(rel_error.max().item()),
    }


def sphere_distance_error(path: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor, eps: float = 1e-7) -> dict[str, float]:
    """Compare discrete spherical path length with analytic great-circle distance."""
    from lrbench.manifolds.sphere import sphere_geodesic_distance, sphere_path_length

    true_distance = sphere_geodesic_distance(z0, z1, eps=eps)
    path_distance = sphere_path_length(path, eps=eps)
    rel_error = torch.abs(path_distance - true_distance) / torch.clamp(true_distance, min=eps)
    return {
        "true_sphere_distance_mean": float(true_distance.mean().item()),
        "path_sphere_distance_mean": float(path_distance.mean().item()),
        "sphere_distance_error_mean": float(rel_error.mean().item()),
        "sphere_distance_error_max": float(rel_error.max().item()),
    }
