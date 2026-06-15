from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import torch

from lrbench.methods.lrw_bvp_solver import normalize_lrw_path
from lrbench.methods.lrw_euclidean_bvp import build_lrw_euclidean_metric


@dataclass
class LRWGeodesicSolverAvailability:
    available: bool
    solver_class_path: str | None
    base_metric_class_path: str | None
    reason: str | None = None


def get_lrw_geodesic_solver_classes() -> tuple[type | None, type | None, LRWGeodesicSolverAvailability]:
    """Return LRW GeodesicSolver and base metric classes when importable."""
    solver_module_name = "lrw.geodesic.solver"
    base_module_name = "lrw.metric.base"
    try:
        solver_module = importlib.import_module(solver_module_name)
        base_module = importlib.import_module(base_module_name)
        solver_cls = getattr(solver_module, "GeodesicSolver")
        base_metric_cls = getattr(base_module, "RiemannianMetric")
        return solver_cls, base_metric_cls, LRWGeodesicSolverAvailability(
            True,
            f"{solver_module_name}.GeodesicSolver",
            f"{base_module_name}.RiemannianMetric",
            None,
        )
    except Exception as exc:  # pragma: no cover - env dependent
        return None, None, LRWGeodesicSolverAvailability(False, None, None, str(exc))


def call_lrw_geodesic_solver_euclidean(
    z0: torch.Tensor,
    z1: torch.Tensor,
    solver_n_steps: int = 100,
    solver_step_size: float = 0.01,
    n_points: int = 32,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
    """Call LRW GeodesicSolver.interpolate and geodesic_distance under Euclidean metric."""
    solver_cls, base_metric_cls, availability = get_lrw_geodesic_solver_classes()
    info: dict[str, Any] = {
        "available": availability.available,
        "solver_class_path": availability.solver_class_path,
        "base_metric_class_path": availability.base_metric_class_path,
        "reason": availability.reason,
    }
    if solver_cls is None or base_metric_cls is None:
        return None, None, info

    metric = build_lrw_euclidean_metric(base_metric_cls)
    solver = solver_cls(metric=metric, n_steps=solver_n_steps, step_size=solver_step_size)

    path = solver.interpolate(z0, z1, n_points=n_points)
    distance = solver.geodesic_distance(z0, z1)
    info.update(
        {
            "metric_class": metric.__class__.__name__,
            "solver_class": solver.__class__.__name__,
            "raw_path_shape": list(path.shape) if isinstance(path, torch.Tensor) else None,
            "raw_distance_shape": list(distance.shape) if isinstance(distance, torch.Tensor) else None,
        }
    )
    return path, distance, info


def normalize_lrw_geodesic_solver_path(raw_path: torch.Tensor, z0_batch: torch.Tensor, z1_batch: torch.Tensor, n_points: int) -> torch.Tensor:
    """Normalize LRW path outputs to [T, B, D]."""
    return normalize_lrw_path(raw_path, z0_batch, z1_batch, requested_n_points=n_points)


def lrw_geodesic_solver_euclidean_metrics(
    path: torch.Tensor,
    distance: torch.Tensor | None,
    z0_batch: torch.Tensor,
    z1_batch: torch.Tensor,
) -> dict[str, Any]:
    """Evaluate LRW GeodesicSolver path and distance against Euclidean ground truth."""
    from lrbench.metrics.accuracy import endpoint_error, euclidean_distance_error
    from lrbench.metrics.geometry import euclidean_path_energy
    from lrbench.metrics.stability import tensor_stability
    from lrbench.methods.euclidean_line import euclidean_line

    metrics: dict[str, Any] = {}
    metrics.update({f"lrw_geodesic_solver_{k}": v for k, v in endpoint_error(path, z0_batch, z1_batch).items()})
    metrics.update({f"lrw_geodesic_solver_{k}": v for k, v in euclidean_distance_error(path, z0_batch, z1_batch).items()})
    metrics.update({f"lrw_geodesic_solver_{k}": v for k, v in tensor_stability(path).items()})

    energy = euclidean_path_energy(path)
    metrics["lrw_geodesic_solver_energy_mean"] = energy["energy_mean"]
    metrics["lrw_geodesic_solver_energy_max"] = energy["energy_max"]
    metrics["lrw_geodesic_solver_energy_min"] = energy["energy_min"]

    straight = euclidean_line(z0_batch, z1_batch, num_points=path.shape[0])
    straight_energy = euclidean_path_energy(straight)
    metrics["straight_euclidean_energy_mean"] = straight_energy["energy_mean"]
    metrics["lrw_geodesic_solver_energy_ratio_over_straight"] = metrics["lrw_geodesic_solver_energy_mean"] / max(straight_energy["energy_mean"], 1e-12)
    metrics["lrw_geodesic_solver_path_shape"] = list(path.shape)
    metrics["lrw_geodesic_solver_num_points_returned"] = int(path.shape[0])

    true_distance = torch.linalg.norm(z1_batch - z0_batch, dim=-1)
    if distance is not None and isinstance(distance, torch.Tensor):
        dist = distance.detach()
        # Normalize common shapes to [B].
        if dist.ndim == 0:
            dist = dist.reshape(1)
        if dist.ndim > 1:
            dist = dist.reshape(-1)
        if dist.numel() == 1 and true_distance.numel() > 1:
            dist = dist.expand_as(true_distance)
        else:
            dist = dist[: true_distance.numel()].reshape_as(true_distance)
        err = torch.abs(dist - true_distance) / torch.clamp(true_distance, min=1e-8)
        metrics["lrw_geodesic_solver_distance_value_mean"] = float(dist.mean().item())
        metrics["lrw_geodesic_solver_true_distance_mean"] = float(true_distance.mean().item())
        metrics["lrw_geodesic_solver_geodesic_distance_error_mean"] = float(err.mean().item())
        metrics["lrw_geodesic_solver_geodesic_distance_error_max"] = float(err.max().item())
        metrics["lrw_geodesic_solver_distance_has_nan"] = bool(torch.isnan(dist).any().item())
        metrics["lrw_geodesic_solver_distance_has_inf"] = bool(torch.isinf(dist).any().item())
    else:
        metrics["lrw_geodesic_solver_geodesic_distance_error_mean"] = None
        metrics["lrw_geodesic_solver_geodesic_distance_error_max"] = None
        metrics["lrw_geodesic_solver_distance_has_nan"] = None
        metrics["lrw_geodesic_solver_distance_has_inf"] = None

    return metrics
