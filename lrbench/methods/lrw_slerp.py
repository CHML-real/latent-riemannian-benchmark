from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Callable

import torch

from lrbench.methods.lrw_bvp_solver import normalize_lrw_path
from lrbench.methods.sphere_great_circle import sphere_great_circle
from lrbench.manifolds.sphere import normalize_to_sphere, radial_unit_error, sphere_geodesic_distance, sphere_path_length


@dataclass
class LRWSlerpAvailability:
    available: bool
    function_path: str | None
    reason: str | None = None


def get_lrw_slerp_function() -> tuple[Callable[..., Any] | None, LRWSlerpAvailability]:
    """Return an LRW slerp path function when importable.

    LRW exposes `slerp_path` at `lrw.geodesic` in v0.3.0, but this helper
    also probes `lrw.geodesic.slerp` so the benchmark remains robust if the
    public export changes.
    """
    candidates = [
        ("lrw.geodesic", "slerp_path"),
        ("lrw.geodesic.slerp", "slerp_path"),
        ("lrw.geodesic.slerp", "slerp"),
        ("lrw.geodesic", "slerp"),
    ]
    errors: list[str] = []
    for module_name, attr_name in candidates:
        try:
            module = importlib.import_module(module_name)
            fn = getattr(module, attr_name)
            if callable(fn):
                return fn, LRWSlerpAvailability(True, f"{module_name}.{attr_name}", None)
        except Exception as exc:  # pragma: no cover - env dependent
            errors.append(f"{module_name}.{attr_name}: {exc}")
    return None, LRWSlerpAvailability(False, None, "; ".join(errors) or "No LRW slerp function was importable.")


def _call_slerp_function(fn: Callable[..., Any], z0: torch.Tensor, z1: torch.Tensor, n_points: int) -> Any:
    """Call LRW slerp function across common signature variants."""
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        kwargs: dict[str, Any] = {}
        if "n_points" in params:
            kwargs["n_points"] = n_points
        elif "num_points" in params:
            kwargs["num_points"] = n_points
        elif "steps" in params:
            kwargs["steps"] = n_points
        elif "n_steps" in params:
            kwargs["n_steps"] = n_points
        if kwargs:
            return fn(z0, z1, **kwargs)
    except Exception:
        # Builtins/partial callables may not expose signatures. Fall through.
        pass

    attempts = [
        lambda: fn(z0, z1, n_points=n_points),
        lambda: fn(z0, z1, num_points=n_points),
        lambda: fn(z0, z1, steps=n_points),
        lambda: fn(z0, z1, n_steps=n_points),
        lambda: fn(z0, z1, n_points),
        lambda: fn(z0, z1),
    ]
    last_exc: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except Exception as exc:  # pragma: no cover - exact LRW signature env-dependent
            last_exc = exc
    raise RuntimeError(f"Could not call LRW slerp function with supported signatures: {last_exc}")


def call_lrw_slerp_path(z0: torch.Tensor, z1: torch.Tensor, n_points: int = 32) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Call LRW slerp path function and return raw output plus info."""
    fn, availability = get_lrw_slerp_function()
    info: dict[str, Any] = {
        "available": availability.available,
        "function_path": availability.function_path,
        "reason": availability.reason,
    }
    if fn is None:
        return None, info
    raw = _call_slerp_function(fn, z0, z1, n_points=n_points)
    if isinstance(raw, tuple):
        raw_path = raw[0]
        info["raw_return_type"] = "tuple"
        if len(raw) > 1:
            info["extra_return_repr"] = repr(raw[1])[:1000]
    else:
        raw_path = raw
        info["raw_return_type"] = type(raw).__name__
    if not isinstance(raw_path, torch.Tensor):
        raise TypeError(f"LRW slerp output is not a Tensor: {type(raw_path)!r}")
    info["raw_path_shape"] = list(raw_path.shape)
    return raw_path, info


def normalize_lrw_slerp_path(raw_path: torch.Tensor, z0_batch: torch.Tensor, z1_batch: torch.Tensor, n_points: int) -> torch.Tensor:
    """Normalize LRW slerp outputs to [T, B, D]."""
    return normalize_lrw_path(raw_path, z0_batch, z1_batch, requested_n_points=n_points)


def lrw_slerp_metrics(path: torch.Tensor, z0_batch: torch.Tensor, z1_batch: torch.Tensor, reference_path: torch.Tensor | None = None) -> dict[str, Any]:
    """Evaluate LRW slerp path against unit-sphere ground truth."""
    from lrbench.metrics.accuracy import endpoint_error
    from lrbench.metrics.stability import tensor_stability

    path = normalize_to_sphere(path)
    z0_batch = normalize_to_sphere(z0_batch)
    z1_batch = normalize_to_sphere(z1_batch)

    metrics: dict[str, Any] = {}
    metrics.update({f"lrw_slerp_{k}": v for k, v in endpoint_error(path, z0_batch, z1_batch).items()})
    metrics.update({f"lrw_slerp_{k}": v for k, v in radial_unit_error(path).items()})
    metrics.update({f"lrw_slerp_{k}": v for k, v in tensor_stability(path).items()})

    true_distance = sphere_geodesic_distance(z0_batch, z1_batch)
    path_distance = sphere_path_length(path)
    err = torch.abs(path_distance - true_distance) / torch.clamp(true_distance, min=1e-8)
    metrics["lrw_slerp_true_sphere_distance_mean"] = float(true_distance.mean().item())
    metrics["lrw_slerp_path_sphere_distance_mean"] = float(path_distance.mean().item())
    metrics["lrw_slerp_sphere_distance_error_mean"] = float(err.mean().item())
    metrics["lrw_slerp_sphere_distance_error_max"] = float(err.max().item())
    metrics["lrw_slerp_path_shape"] = list(path.shape)
    metrics["lrw_slerp_num_points_returned"] = int(path.shape[0])

    if reference_path is None:
        reference_path = sphere_great_circle(z0_batch, z1_batch, num_points=path.shape[0])
    if reference_path.shape == path.shape:
        abs_err = torch.abs(path - reference_path)
        denom = torch.linalg.norm(reference_path.reshape(reference_path.shape[0], -1), dim=-1).clamp_min(1e-12)
        rel = torch.linalg.norm((path - reference_path).reshape(path.shape[0], -1), dim=-1) / denom
        metrics["lrw_slerp_vs_reference_same_shape"] = True
        metrics["lrw_slerp_vs_reference_mean_abs_error"] = float(abs_err.mean().item())
        metrics["lrw_slerp_vs_reference_max_abs_error"] = float(abs_err.max().item())
        metrics["lrw_slerp_vs_reference_relative_error_mean"] = float(rel.mean().item())
        metrics["lrw_slerp_vs_reference_relative_error_max"] = float(rel.max().item())
    else:
        metrics["lrw_slerp_vs_reference_same_shape"] = False
        metrics["lrw_slerp_vs_reference_mean_abs_error"] = None
        metrics["lrw_slerp_vs_reference_max_abs_error"] = None
        metrics["lrw_slerp_vs_reference_relative_error_mean"] = None
        metrics["lrw_slerp_vs_reference_relative_error_max"] = None

    return metrics
