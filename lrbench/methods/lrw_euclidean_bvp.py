from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import torch

from lrbench.methods.lrw_bvp_solver import normalize_lrw_path


@dataclass
class LRWEuclideanBVPAvailability:
    available: bool
    bvp_class_path: str | None
    base_metric_class_path: str | None
    reason: str | None = None


def get_lrw_euclidean_bvp_classes() -> tuple[type | None, type | None, LRWEuclideanBVPAvailability]:
    """Return LRW BVPSolver and base metric classes when importable."""
    bvp_module_name = "lrw.geodesic.bvp"
    base_module_name = "lrw.metric.base"
    try:
        bvp_module = importlib.import_module(bvp_module_name)
        base_module = importlib.import_module(base_module_name)
        bvp_cls = getattr(bvp_module, "BVPSolver")
        base_metric_cls = getattr(base_module, "RiemannianMetric")
        return bvp_cls, base_metric_cls, LRWEuclideanBVPAvailability(
            True,
            f"{bvp_module_name}.BVPSolver",
            f"{base_module_name}.RiemannianMetric",
            None,
        )
    except Exception as exc:  # pragma: no cover - env dependent
        return None, None, LRWEuclideanBVPAvailability(False, None, None, str(exc))


def build_lrw_euclidean_metric(base_metric_cls: type) -> Any:
    """Build a minimal Euclidean metric compatible with LRW solvers.

    The class is defined dynamically so it can inherit LRW's RiemannianMetric
    when available while remaining easy to mock in tests.
    """

    class LRBenchEuclideanMetric(base_metric_cls):  # type: ignore[misc, valid-type]
        def metric_tensor(self, z: torch.Tensor) -> torch.Tensor:
            flat = z.reshape(-1, z.shape[-1])
            eye = torch.eye(z.shape[-1], device=z.device, dtype=z.dtype)
            return eye.expand(flat.shape[0], z.shape[-1], z.shape[-1]).clone()

        def christoffel(self, z: torch.Tensor) -> torch.Tensor:
            flat = z.reshape(-1, z.shape[-1])
            dim = z.shape[-1]
            return torch.zeros(flat.shape[0], dim, dim, dim, device=z.device, dtype=z.dtype)

        def geodesic_acceleration(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(v)

    return LRBenchEuclideanMetric()


def call_lrw_euclidean_bvp_geodesic_path(
    z0: torch.Tensor,
    z1: torch.Tensor,
    solver_n_steps: int = 20,
    solver_step_size: float = 0.05,
    solver_lr: float = 0.1,
    solver_max_iter: int = 50,
    solver_tol: float = 1e-3,
    n_points: int = 32,
) -> tuple[torch.Tensor | None, dict[str, Any], dict[str, Any]]:
    """Call LRW BVPSolver.geodesic_path under a Euclidean metric."""
    bvp_cls, base_metric_cls, availability = get_lrw_euclidean_bvp_classes()
    info: dict[str, Any] = {
        "available": availability.available,
        "bvp_class_path": availability.bvp_class_path,
        "base_metric_class_path": availability.base_metric_class_path,
        "reason": availability.reason,
    }
    if bvp_cls is None or base_metric_cls is None:
        return None, info, {}

    metric = build_lrw_euclidean_metric(base_metric_cls)
    solver = bvp_cls(
        metric=metric,
        n_steps=solver_n_steps,
        step_size=solver_step_size,
        lr=solver_lr,
        max_iter=solver_max_iter,
        tol=solver_tol,
    )
    output = solver.geodesic_path(z0, z1, n_points=n_points)
    if isinstance(output, tuple) and len(output) >= 2:
        raw_path, solver_info = output[0], output[1]
    else:
        raw_path, solver_info = output, {}
    if not isinstance(solver_info, dict):
        solver_info = {"raw_solver_info": str(solver_info)}
    info.update(
        {
            "metric_class": metric.__class__.__name__,
            "solver_class": solver.__class__.__name__,
            "raw_path_shape": list(raw_path.shape) if isinstance(raw_path, torch.Tensor) else None,
        }
    )
    return raw_path, info, solver_info


def lrw_euclidean_bvp_path_metrics(
    path: torch.Tensor,
    z0_batch: torch.Tensor,
    z1_batch: torch.Tensor,
) -> dict[str, Any]:
    """Endpoint, distance, stability, energy, and straight-line comparison."""
    from lrbench.metrics.accuracy import endpoint_error, euclidean_distance_error
    from lrbench.metrics.geometry import euclidean_path_energy
    from lrbench.metrics.stability import tensor_stability
    from lrbench.methods.euclidean_line import euclidean_line

    metrics: dict[str, Any] = {}
    metrics.update({f"lrw_euclidean_bvp_{k}": v for k, v in endpoint_error(path, z0_batch, z1_batch).items()})
    metrics.update({f"lrw_euclidean_bvp_{k}": v for k, v in euclidean_distance_error(path, z0_batch, z1_batch).items()})
    metrics.update({f"lrw_euclidean_bvp_{k}": v for k, v in tensor_stability(path).items()})
    energy = euclidean_path_energy(path)
    metrics["lrw_euclidean_bvp_energy_mean"] = energy["energy_mean"]
    metrics["lrw_euclidean_bvp_energy_max"] = energy["energy_max"]
    metrics["lrw_euclidean_bvp_energy_min"] = energy["energy_min"]

    straight = euclidean_line(z0_batch, z1_batch, num_points=path.shape[0])
    straight_energy = euclidean_path_energy(straight)
    metrics["straight_euclidean_energy_mean"] = straight_energy["energy_mean"]
    metrics["lrw_euclidean_bvp_energy_ratio_over_straight"] = metrics["lrw_euclidean_bvp_energy_mean"] / max(straight_energy["energy_mean"], 1e-12)
    metrics["lrw_euclidean_bvp_path_shape"] = list(path.shape)
    metrics["lrw_euclidean_bvp_num_points_returned"] = int(path.shape[0])
    return metrics


def normalize_lrw_euclidean_bvp_path(raw_path: torch.Tensor, z0_batch: torch.Tensor, z1_batch: torch.Tensor, n_points: int) -> torch.Tensor:
    return normalize_lrw_path(raw_path, z0_batch, z1_batch, requested_n_points=n_points)
