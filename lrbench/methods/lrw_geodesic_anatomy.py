from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import torch

from lrbench.methods.euclidean_line import euclidean_line
from lrbench.methods.lrw_euclidean_bvp import build_lrw_euclidean_metric
from lrbench.metrics.accuracy import endpoint_error, euclidean_distance_error
from lrbench.metrics.geometry import euclidean_path_energy
from lrbench.metrics.stability import tensor_stability


@dataclass
class LRWGeodesicAnatomyAvailability:
    available: bool
    reason: str | None = None
    geodesic_solver_class_path: str | None = None
    bvp_solver_class_path: str | None = None
    base_metric_class_path: str | None = None


def get_lrw_solver_classes() -> tuple[type | None, type | None, type | None, LRWGeodesicAnatomyAvailability]:
    try:
        geodesic_module = importlib.import_module("lrw.geodesic.solver")
        bvp_module = importlib.import_module("lrw.geodesic.bvp")
        base_module = importlib.import_module("lrw.metric.base")
        geodesic_cls = getattr(geodesic_module, "GeodesicSolver")
        bvp_cls = getattr(bvp_module, "BVPSolver")
        base_metric_cls = getattr(base_module, "RiemannianMetric")
        return geodesic_cls, bvp_cls, base_metric_cls, LRWGeodesicAnatomyAvailability(
            available=True,
            reason=None,
            geodesic_solver_class_path="lrw.geodesic.solver.GeodesicSolver",
            bvp_solver_class_path="lrw.geodesic.bvp.BVPSolver",
            base_metric_class_path="lrw.metric.base.RiemannianMetric",
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, None, None, LRWGeodesicAnatomyAvailability(available=False, reason=str(exc))


def _jsonify(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        v = value.detach().cpu()
        if v.numel() <= 32:
            return v.tolist()
        return {"shape": list(v.shape), "mean": float(v.float().mean().item())}
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def unpack_lrw_output(output: Any) -> tuple[Any, dict[str, Any]]:
    """Unpack common LRW outputs while preserving diagnostic info."""
    info: dict[str, Any] = {"raw_output_type": type(output).__name__}
    if isinstance(output, tuple):
        info["tuple_len"] = len(output)
        if len(output) >= 1:
            first = output[0]
            rest = output[1:]
            info["tuple_rest"] = _jsonify(rest)
            if len(output) >= 2 and isinstance(output[1], dict):
                info["solver_info"] = _jsonify(output[1])
            return first, info
    return output, info


def tensor_preview(t: torch.Tensor | None, max_items: int = 8) -> Any:
    if t is None or not isinstance(t, torch.Tensor):
        return None
    v = t.detach().cpu()
    if v.numel() <= max_items:
        return v.tolist()
    flat = v.reshape(-1)[:max_items]
    return {"shape": list(v.shape), "head": flat.tolist()}


def path_axis_candidates(raw_path: Any, z0: torch.Tensor, z1: torch.Tensor, requested_n_points: int) -> list[dict[str, Any]]:
    """Generate plausible [T,B,D] interpretations for an LRW path tensor."""
    if not isinstance(raw_path, torch.Tensor):
        return []
    x = raw_path.detach()
    b, d = int(z0.shape[0]), int(z0.shape[-1])
    candidates: list[tuple[str, torch.Tensor]] = []

    def add(name: str, p: torch.Tensor) -> None:
        if p.ndim == 3 and p.shape[-1] == d:
            candidates.append((name, p))

    if x.ndim == 3:
        add("as_TBD", x)
        add("as_BTD_transpose_0_1", x.transpose(0, 1))
        if x.shape[1] == d:
            add("as_TDB_permute_0_2_1", x.permute(0, 2, 1))
        if x.shape[0] == d:
            add("as_DBT_permute_2_1_0", x.permute(2, 1, 0))
        if x.shape[2] == requested_n_points:
            add("as_BDT_permute_2_0_1", x.permute(2, 0, 1))
    elif x.ndim == 2:
        if x.shape[-1] == d:
            add("as_TD_unsqueeze_batch", x.unsqueeze(1))
        if x.shape[0] == d:
            add("as_DT_transpose_unsqueeze_batch", x.transpose(0, 1).unsqueeze(1))
    elif x.ndim == 1 and d == 1:
        add("as_T_unsqueeze_batch_dim", x.reshape(-1, 1, 1))

    # Deduplicate by name+shape+first few values.
    seen: set[tuple[str, tuple[int, ...]]] = set()
    out: list[dict[str, Any]] = []
    for name, p in candidates:
        key = (name, tuple(int(v) for v in p.shape))
        if key in seen:
            continue
        seen.add(key)
        out.append(evaluate_path_candidate(name, p, z0, z1))
    return out


def evaluate_path_candidate(name: str, path: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor) -> dict[str, Any]:
    """Evaluate one [T,B,D] candidate without assuming it is correct."""
    p = path.detach()
    info: dict[str, Any] = {
        "candidate_name": name,
        "shape": list(p.shape),
        "first_point": tensor_preview(p[0]),
        "last_point": tensor_preview(p[-1]),
    }
    if p.ndim != 3 or p.shape[-1] != z0.shape[-1]:
        info.update({"valid_shape": False, "reason": "not [T,B,D] with matching D"})
        return info
    # Match batch conservatively.
    if p.shape[1] != z0.shape[0]:
        if p.shape[1] == 1:
            z0_eval = z0[:1]
            z1_eval = z1[:1]
        else:
            info.update({"valid_shape": False, "reason": "batch dimension does not match endpoints"})
            return info
    else:
        z0_eval = z0
        z1_eval = z1

    endpoint = endpoint_error(p, z0_eval, z1_eval)
    distance = euclidean_distance_error(p, z0_eval, z1_eval)
    stability = tensor_stability(p)
    energy = euclidean_path_energy(p)
    straight = euclidean_line(z0_eval, z1_eval, num_points=p.shape[0])
    straight_energy = euclidean_path_energy(straight)["energy_mean"]
    step = p[1:] - p[:-1] if p.shape[0] > 1 else torch.zeros_like(p[:0])
    step_norm = torch.linalg.norm(step, dim=-1) if step.numel() else torch.zeros(1, device=p.device, dtype=p.dtype)
    total_len = step_norm.sum(dim=0) if step_norm.ndim > 1 else step_norm
    norms = torch.linalg.norm(p, dim=-1)

    info.update(
        {
            "valid_shape": True,
            "endpoint_error_start_mean": endpoint["endpoint_error_start_mean"],
            "endpoint_error_end_mean": endpoint["endpoint_error_end_mean"],
            "endpoint_error_mean": endpoint["endpoint_error_mean"],
            "endpoint_error_max": endpoint["endpoint_error_max"],
            "distance_error_mean": distance["distance_error_mean"],
            "distance_error_max": distance["distance_error_max"],
            "energy_mean": energy["energy_mean"],
            "straight_energy_mean": straight_energy,
            "energy_ratio_over_straight": energy["energy_mean"] / max(straight_energy, 1e-12),
            "step_length_mean": float(step_norm.mean().item()) if step_norm.numel() else 0.0,
            "step_length_max": float(step_norm.max().item()) if step_norm.numel() else 0.0,
            "total_path_length_mean": float(total_len.mean().item()) if total_len.numel() else 0.0,
            "norm_mean": float(norms.mean().item()),
            "norm_min": float(norms.min().item()),
            "norm_max": float(norms.max().item()),
            "has_nan": stability["has_nan"],
            "has_inf": stability["has_inf"],
        }
    )
    return info


def call_lrw_geodesic_outputs(
    z0: torch.Tensor,
    z1: torch.Tensor,
    n_points: int,
    geodesic_n_steps: int = 100,
    geodesic_step_size: float = 0.01,
    bvp_n_steps: int = 20,
    bvp_step_size: float = 0.05,
    bvp_lr: float = 0.1,
    bvp_max_iter: int = 50,
    bvp_tol: float = 0.001,
) -> dict[str, Any]:
    geodesic_cls, bvp_cls, base_metric_cls, availability = get_lrw_solver_classes()
    result: dict[str, Any] = {
        "availability": _jsonify(availability.__dict__),
        "calls": [],
    }
    if not availability.available or geodesic_cls is None or bvp_cls is None or base_metric_cls is None:
        return result

    metric = build_lrw_euclidean_metric(base_metric_cls)

    # GeodesicSolver.interpolate
    try:
        solver = geodesic_cls(metric=metric, n_steps=geodesic_n_steps, step_size=geodesic_step_size)
        out = solver.interpolate(z0, z1, n_points=n_points)
        raw, unpack_info = unpack_lrw_output(out)
        call = {
            "call_name": "GeodesicSolver.interpolate",
            "status": "success",
            "unpack": unpack_info,
            "raw_type": type(raw).__name__,
            "raw_shape": list(raw.shape) if isinstance(raw, torch.Tensor) else None,
            "raw_preview": tensor_preview(raw) if isinstance(raw, torch.Tensor) else _jsonify(raw),
            "axis_candidates": path_axis_candidates(raw, z0, z1, n_points),
        }
    except Exception as exc:  # pragma: no cover
        call = {"call_name": "GeodesicSolver.interpolate", "status": "failure", "error": str(exc)}
    result["calls"].append(call)

    # GeodesicSolver.geodesic_distance
    try:
        out = solver.geodesic_distance(z0, z1)
        raw, unpack_info = unpack_lrw_output(out)
        true_distance = torch.linalg.norm(z1 - z0, dim=-1)
        distance_info: dict[str, Any] = {
            "call_name": "GeodesicSolver.geodesic_distance",
            "status": "success",
            "unpack": unpack_info,
            "raw_type": type(raw).__name__,
            "raw_shape": list(raw.shape) if isinstance(raw, torch.Tensor) else None,
            "raw_value": tensor_preview(raw) if isinstance(raw, torch.Tensor) else _jsonify(raw),
            "true_distance": tensor_preview(true_distance),
        }
        if isinstance(raw, torch.Tensor):
            dist = raw.detach().reshape(-1)
            true = true_distance.detach().reshape(-1)
            if dist.numel() == 1 and true.numel() > 1:
                dist = dist.expand_as(true)
            dist = dist[: true.numel()].reshape_as(true)
            rel = torch.abs(dist - true) / torch.clamp(true, min=1e-8)
            distance_info["relative_error_mean"] = float(rel.mean().item())
            distance_info["relative_error_max"] = float(rel.max().item())
            distance_info["distance_value_mean"] = float(dist.mean().item())
            distance_info["true_distance_mean"] = float(true.mean().item())
    except Exception as exc:  # pragma: no cover
        distance_info = {"call_name": "GeodesicSolver.geodesic_distance", "status": "failure", "error": str(exc)}
    result["calls"].append(distance_info)

    # BVPSolver.geodesic_path
    try:
        bvp = bvp_cls(metric=metric, n_steps=bvp_n_steps, step_size=bvp_step_size, lr=bvp_lr, max_iter=bvp_max_iter, tol=bvp_tol)
        out = bvp.geodesic_path(z0, z1, n_points=n_points)
        raw, unpack_info = unpack_lrw_output(out)
        call = {
            "call_name": "BVPSolver.geodesic_path",
            "status": "success",
            "unpack": unpack_info,
            "raw_type": type(raw).__name__,
            "raw_shape": list(raw.shape) if isinstance(raw, torch.Tensor) else None,
            "raw_preview": tensor_preview(raw) if isinstance(raw, torch.Tensor) else _jsonify(raw),
            "axis_candidates": path_axis_candidates(raw, z0, z1, n_points),
        }
    except Exception as exc:  # pragma: no cover
        call = {"call_name": "BVPSolver.geodesic_path", "status": "failure", "error": str(exc)}
    result["calls"].append(call)

    # BVPSolver.solve
    try:
        bvp = bvp_cls(metric=metric, n_steps=bvp_n_steps, step_size=bvp_step_size, lr=bvp_lr, max_iter=bvp_max_iter, tol=bvp_tol)
        out = bvp.solve(z0, z1)
        raw, unpack_info = unpack_lrw_output(out)
        call = {
            "call_name": "BVPSolver.solve",
            "status": "success",
            "unpack": unpack_info,
            "raw_type": type(raw).__name__,
            "raw_shape": list(raw.shape) if isinstance(raw, torch.Tensor) else None,
            "raw_preview": tensor_preview(raw) if isinstance(raw, torch.Tensor) else _jsonify(raw),
            "axis_candidates": path_axis_candidates(raw, z0, z1, n_points) if isinstance(raw, torch.Tensor) else [],
        }
    except Exception as exc:  # pragma: no cover
        call = {"call_name": "BVPSolver.solve", "status": "failure", "error": str(exc)}
    result["calls"].append(call)

    return result


def summarize_anatomy(calls: list[dict[str, Any]], endpoint_tolerance: float = 1e-3) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for call in calls:
        for cand in call.get("axis_candidates", []) or []:
            if cand.get("valid_shape"):
                item = dict(cand)
                item["call_name"] = call.get("call_name")
                candidates.append(item)
    best = min(candidates, key=lambda c: float(c.get("endpoint_error_mean", float("inf")))) if candidates else None
    valid = [c for c in candidates if float(c.get("endpoint_error_mean", float("inf"))) <= endpoint_tolerance]
    geodesic_distance = next((c for c in calls if c.get("call_name") == "GeodesicSolver.geodesic_distance"), {})
    successful_calls = [c for c in calls if c.get("status") == "success"]
    return {
        "anatomy_call_count": len(calls),
        "anatomy_successful_call_count": len(successful_calls),
        "anatomy_axis_candidate_count": len(candidates),
        "anatomy_axis_endpoint_valid_count": len(valid),
        "anatomy_best_axis_endpoint_error": None if best is None else best.get("endpoint_error_mean"),
        "anatomy_best_axis_name": None if best is None else best.get("candidate_name"),
        "anatomy_best_axis_call": None if best is None else best.get("call_name"),
        "anatomy_best_axis_distance_error": None if best is None else best.get("distance_error_mean"),
        "anatomy_best_axis_energy_ratio": None if best is None else best.get("energy_ratio_over_straight"),
        "anatomy_geodesic_distance_relative_error_mean": geodesic_distance.get("relative_error_mean"),
        "anatomy_verdict": "axis_interpretation_pass" if valid else "no_axis_interpretation_pass",
    }
