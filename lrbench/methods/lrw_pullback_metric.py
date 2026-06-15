from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Any

import torch

from lrbench.manifolds.pullback import toy_pullback_metric


@dataclass
class LRWPullbackAvailability:
    available: bool
    class_path: str | None
    reason: str | None = None


def get_lrw_pullback_metric_class() -> tuple[type | None, LRWPullbackAvailability]:
    """Return lrw.metric.pullback.PullbackMetric if importable."""
    module_name = "lrw.metric.pullback"
    class_name = "PullbackMetric"
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls, LRWPullbackAvailability(True, f"{module_name}.{class_name}", None)
    except Exception as exc:  # pragma: no cover - exact import error depends on env
        return None, LRWPullbackAvailability(False, None, str(exc))


def compute_lrw_pullback_metric(
    z: torch.Tensor,
    decoder: Callable[[torch.Tensor], torch.Tensor],
    regularization: float = 1e-5,
    chunk_size: int | None = None,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Compute LRW PullbackMetric.metric_tensor(z), or return metadata if unavailable."""
    cls, availability = get_lrw_pullback_metric_class()
    info: dict[str, Any] = {
        "available": availability.available,
        "class_path": availability.class_path,
        "reason": availability.reason,
    }
    if cls is None:
        return None, info

    metric = cls(decoder=decoder, chunk_size=chunk_size, regularization=regularization)
    G = metric.metric_tensor(z)
    info["metric_class"] = metric.__class__.__name__
    info["metric_tensor_shape"] = list(G.shape)
    return G, info


def metric_tensor_stats(G: torch.Tensor, prefix: str = "lrw_pullback") -> dict[str, float | bool | list[int]]:
    """Basic SPD/stability diagnostics for a metric tensor batch."""
    if G.ndim < 2 or G.shape[-1] != G.shape[-2]:
        raise ValueError(f"expected metric tensor shape [..., d, d], got {tuple(G.shape)}")

    eigvals = torch.linalg.eigvalsh(G)
    min_eig = eigvals[..., 0]
    max_eig = eigvals[..., -1]
    cond = max_eig / torch.clamp(min_eig, min=1e-12)
    return {
        f"{prefix}_shape": list(G.shape),
        f"{prefix}_min_eigenvalue_mean": float(min_eig.mean().item()),
        f"{prefix}_min_eigenvalue_min": float(min_eig.min().item()),
        f"{prefix}_max_eigenvalue_mean": float(max_eig.mean().item()),
        f"{prefix}_condition_number_mean": float(cond.mean().item()),
        f"{prefix}_condition_number_max": float(cond.max().item()),
        f"{prefix}_spd_violation_rate": float((min_eig <= 0).float().mean().item()),
        f"{prefix}_has_nan": bool(torch.isnan(G).any().item()),
        f"{prefix}_has_inf": bool(torch.isinf(G).any().item()),
        f"{prefix}_nan_rate": float(torch.isnan(G).float().mean().item()),
        f"{prefix}_inf_rate": float(torch.isinf(G).float().mean().item()),
    }


def compare_metric_tensors(
    lrw_G: torch.Tensor,
    ref_G: torch.Tensor,
    prefix: str = "lrw_vs_reference",
) -> dict[str, float | bool]:
    """Compare LRW PullbackMetric against the benchmark's reference implementation."""
    if lrw_G.shape != ref_G.shape:
        return {
            f"{prefix}_same_shape": False,
            f"{prefix}_mean_abs_error": float("nan"),
            f"{prefix}_max_abs_error": float("nan"),
            f"{prefix}_relative_frobenius_error": float("nan"),
        }

    diff = lrw_G - ref_G
    diff_norm = torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1), dim=-1)
    ref_norm = torch.linalg.vector_norm(ref_G.reshape(ref_G.shape[0], -1), dim=-1).clamp_min(1e-12)
    rel = diff_norm / ref_norm
    return {
        f"{prefix}_same_shape": True,
        f"{prefix}_mean_abs_error": float(diff.abs().mean().item()),
        f"{prefix}_max_abs_error": float(diff.abs().max().item()),
        f"{prefix}_relative_frobenius_error_mean": float(rel.mean().item()),
        f"{prefix}_relative_frobenius_error_max": float(rel.max().item()),
    }


def compute_reference_pullback_metric(
    z: torch.Tensor,
    beta: float = 6.0,
    sigma: float = 0.7,
    cross: float = 0.2,
    damping: float = 1e-5,
) -> torch.Tensor:
    """Reference metric using lrbench's own toy pullback implementation."""
    return toy_pullback_metric(z, beta=beta, sigma=sigma, cross=cross, damping=damping)
