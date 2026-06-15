from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

import torch

from lrbench.methods.lrw_pullback_metric import compute_reference_pullback_metric
from lrbench.metrics.pullback_energy import pullback_path_energy, pullback_path_length


@dataclass
class LRWBVPAvailability:
    available: bool
    bvp_class_path: str | None
    metric_class_path: str | None
    reason: str | None = None


def get_lrw_bvp_classes() -> tuple[type | None, type | None, LRWBVPAvailability]:
    """Return LRW BVPSolver and PullbackMetric classes when importable."""
    bvp_module_name = "lrw.geodesic.bvp"
    metric_module_name = "lrw.metric.pullback"
    try:
        bvp_module = importlib.import_module(bvp_module_name)
        metric_module = importlib.import_module(metric_module_name)
        bvp_cls = getattr(bvp_module, "BVPSolver")
        metric_cls = getattr(metric_module, "PullbackMetric")
        return bvp_cls, metric_cls, LRWBVPAvailability(
            True,
            f"{bvp_module_name}.BVPSolver",
            f"{metric_module_name}.PullbackMetric",
            None,
        )
    except Exception as exc:  # pragma: no cover - env dependent
        return None, None, LRWBVPAvailability(False, None, None, str(exc))


def normalize_lrw_path(
    raw_path: torch.Tensor,
    z0_batch: torch.Tensor,
    z1_batch: torch.Tensor,
    requested_n_points: int | None = None,
) -> torch.Tensor:
    """Normalize LRW path outputs to [num_points, batch, dim].

    LRW versions or custom solvers may return [T, D], [T, B, D], or [B, T, D].
    This benchmark uses [T, B, D] internally.
    """
    path = raw_path if isinstance(raw_path, torch.Tensor) else torch.as_tensor(raw_path, device=z0_batch.device, dtype=z0_batch.dtype)
    if path.ndim == 1:
        path = path.view(1, 1, -1)
    elif path.ndim == 2:
        # [T, D] for single item or [B, D] degenerate output.
        if path.shape[-1] != z0_batch.shape[-1]:
            raise ValueError(f"cannot normalize path shape {tuple(path.shape)} for latent dim {z0_batch.shape[-1]}")
        path = path.unsqueeze(1)
    elif path.ndim == 3:
        # [T, B, D] expected, or [B, T, D].
        if path.shape[-1] != z0_batch.shape[-1]:
            raise ValueError(f"cannot normalize path shape {tuple(path.shape)} for latent dim {z0_batch.shape[-1]}")
        batch_size = z0_batch.shape[0]
        # If requested_n_points disambiguates [B, T, D], prefer it.
        if requested_n_points is not None and path.shape[0] == batch_size and path.shape[1] == requested_n_points:
            path = path.transpose(0, 1).contiguous()
        elif path.shape[1] == batch_size:
            pass
        elif path.shape[0] == batch_size:
            path = path.transpose(0, 1).contiguous()
        elif batch_size == 1:
            # Ambiguous [T, ?, D], keep as-is if the second axis is not batch.
            pass
        else:
            raise ValueError(f"cannot infer batch axis for path shape {tuple(path.shape)} and batch {batch_size}")
    else:
        raise ValueError(f"expected path ndim <= 3, got shape {tuple(path.shape)}")

    if requested_n_points is not None and path.shape[0] != requested_n_points:
        # Not a hard failure. Some solvers return internal n_steps+1 instead of requested n_points.
        path = path.contiguous()
    return path


def compute_lrw_bvp_geodesic_path(
    z0: torch.Tensor,
    z1: torch.Tensor,
    decoder: Callable[[torch.Tensor], torch.Tensor],
    regularization: float = 1e-5,
    chunk_size: int | None = None,
    solver_n_steps: int = 20,
    solver_step_size: float = 0.05,
    solver_lr: float = 0.1,
    solver_max_iter: int = 50,
    solver_tol: float = 1e-3,
    n_points: int = 32,
) -> tuple[torch.Tensor | None, dict[str, Any], dict[str, Any]]:
    """Call LRW PullbackMetric + BVPSolver.geodesic_path."""
    bvp_cls, metric_cls, availability = get_lrw_bvp_classes()
    info: dict[str, Any] = {
        "available": availability.available,
        "bvp_class_path": availability.bvp_class_path,
        "metric_class_path": availability.metric_class_path,
        "reason": availability.reason,
    }
    if bvp_cls is None or metric_cls is None:
        return None, info, {}

    metric = metric_cls(decoder=decoder, chunk_size=chunk_size, regularization=regularization)
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

    info.update(
        {
            "metric_class": metric.__class__.__name__,
            "solver_class": solver.__class__.__name__,
            "raw_path_shape": list(raw_path.shape) if isinstance(raw_path, torch.Tensor) else None,
        }
    )
    if not isinstance(solver_info, dict):
        solver_info = {"raw_solver_info": str(solver_info)}
    return raw_path, info, solver_info


def lrw_bvp_path_metrics(
    path: torch.Tensor,
    z0_batch: torch.Tensor,
    z1_batch: torch.Tensor,
    beta: float = 6.0,
    sigma: float = 0.7,
    cross: float = 0.2,
    regularization: float = 1e-5,
) -> dict[str, Any]:
    """Endpoint/stability/energy diagnostics for a normalized LRW BVP path."""
    from lrbench.metrics.accuracy import endpoint_error, euclidean_distance_error
    from lrbench.metrics.stability import tensor_stability

    metrics: dict[str, Any] = {}
    metrics.update({f"lrw_bvp_{k}": v for k, v in endpoint_error(path, z0_batch, z1_batch).items()})
    metrics.update({f"lrw_bvp_{k}": v for k, v in euclidean_distance_error(path, z0_batch, z1_batch).items()})
    metrics.update({f"lrw_bvp_{k}": v for k, v in tensor_stability(path).items()})
    metrics.update({f"lrw_bvp_{k}": v for k, v in pullback_path_energy(path, beta=beta, sigma=sigma, cross=cross, damping=regularization).items()})
    metrics.update({f"lrw_bvp_{k}": v for k, v in pullback_path_length(path, beta=beta, sigma=sigma, cross=cross, damping=regularization).items()})
    metrics["lrw_bvp_path_shape"] = list(path.shape)
    metrics["lrw_bvp_num_points_returned"] = int(path.shape[0])
    return metrics
