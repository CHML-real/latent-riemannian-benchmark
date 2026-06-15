from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lrbench.methods.lrw_euclidean_bvp import build_lrw_euclidean_metric
from lrbench.methods.lrw_geodesic_solver import get_lrw_geodesic_solver_classes, normalize_lrw_geodesic_solver_path


@dataclass
class ScaleProbeCase:
    case_id: str
    n_steps: int
    step_size: float


def make_scale_probe_cases(n_steps_values: list[int], step_size_values: list[float]) -> list[ScaleProbeCase]:
    cases: list[ScaleProbeCase] = []
    idx = 1
    for n_steps in n_steps_values:
        for step_size in step_size_values:
            cases.append(ScaleProbeCase(f"case_{idx:03d}", int(n_steps), float(step_size)))
            idx += 1
    return cases


def _as_batch_distance(distance: torch.Tensor, true_distance: torch.Tensor) -> torch.Tensor:
    dist = distance.detach()
    if dist.ndim == 0:
        dist = dist.reshape(1)
    if dist.ndim > 1:
        dist = dist.reshape(-1)
    if dist.numel() == 1 and true_distance.numel() > 1:
        dist = dist.expand_as(true_distance)
    else:
        dist = dist[: true_distance.numel()].reshape_as(true_distance)
    return dist


def _path_stats(path: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor) -> dict[str, Any]:
    true_distance = torch.linalg.norm(z1 - z0, dim=-1)
    start_error = torch.linalg.norm(path[0] - z0, dim=-1)
    end_error = torch.linalg.norm(path[-1] - z1, dim=-1)
    step_lengths = torch.linalg.norm(path[1:] - path[:-1], dim=-1) if path.shape[0] >= 2 else torch.zeros_like(true_distance)
    total_length = step_lengths.sum(dim=0) if step_lengths.ndim >= 2 else step_lengths.reshape_as(true_distance)
    displacement = torch.linalg.norm(path[-1] - path[0], dim=-1)
    safe_true = torch.clamp(true_distance, min=1e-12)
    return {
        "endpoint_error_start_mean": float(start_error.mean().item()),
        "endpoint_error_end_mean": float(end_error.mean().item()),
        "endpoint_error_mean": float(((start_error + end_error) * 0.5).mean().item()),
        "endpoint_error_max": float(torch.maximum(start_error, end_error).max().item()),
        "total_path_length_mean": float(total_length.mean().item()),
        "displacement_mean": float(displacement.mean().item()),
        "true_distance_mean": float(true_distance.mean().item()),
        "movement_ratio": float((total_length / safe_true).mean().item()),
        "displacement_ratio": float((displacement / safe_true).mean().item()),
        "step_length_mean": float(step_lengths.mean().item()) if step_lengths.numel() else 0.0,
        "step_length_max": float(step_lengths.max().item()) if step_lengths.numel() else 0.0,
        "has_nan": bool(torch.isnan(path).any().item()),
        "has_inf": bool(torch.isinf(path).any().item()),
    }


def run_lrw_solver_scale_probe(
    z0: torch.Tensor,
    z1: torch.Tensor,
    n_points: int,
    n_steps_values: list[int],
    step_size_values: list[float],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Probe how LRW GeodesicSolver outputs scale with n_steps and step_size."""
    solver_cls, base_metric_cls, availability = get_lrw_geodesic_solver_classes()
    lrw_info: dict[str, Any] = {
        "available": availability.available,
        "solver_class_path": availability.solver_class_path,
        "base_metric_class_path": availability.base_metric_class_path,
        "reason": availability.reason,
    }
    if solver_cls is None or base_metric_cls is None:
        return [], lrw_info, {"scale_probe_case_count": 0, "scale_probe_success_count": 0}

    metric = build_lrw_euclidean_metric(base_metric_cls)
    cases = make_scale_probe_cases(n_steps_values, step_size_values)
    true_distance = torch.linalg.norm(z1 - z0, dim=-1)
    rows: list[dict[str, Any]] = []

    for case in cases:
        row: dict[str, Any] = {
            "case_id": case.case_id,
            "n_steps": case.n_steps,
            "step_size": case.step_size,
            "n_steps_times_step_size": float(case.n_steps * case.step_size),
        }
        try:
            solver = solver_cls(metric=metric, n_steps=case.n_steps, step_size=case.step_size)
            raw_path = solver.interpolate(z0, z1, n_points=n_points)
            raw_distance = solver.geodesic_distance(z0, z1)
            path = normalize_lrw_geodesic_solver_path(raw_path, z0, z1, n_points=n_points)
            dist = _as_batch_distance(raw_distance, true_distance)
            stats = _path_stats(path, z0, z1)
            safe_true = torch.clamp(true_distance, min=1e-12)
            distance_ratio = float((dist / safe_true).mean().item())
            row.update(
                {
                    "status": "success",
                    "raw_path_shape": list(raw_path.shape) if isinstance(raw_path, torch.Tensor) else None,
                    "raw_distance_shape": list(raw_distance.shape) if isinstance(raw_distance, torch.Tensor) else None,
                    "geodesic_distance_mean": float(dist.mean().item()),
                    "true_distance_mean": float(true_distance.mean().item()),
                    "distance_ratio": distance_ratio,
                    "distance_ratio_over_step_size": distance_ratio / max(case.step_size, 1e-12),
                    "distance_ratio_over_n_steps_step_size": distance_ratio / max(case.n_steps * case.step_size, 1e-12),
                    "movement_ratio_over_step_size": stats["movement_ratio"] / max(case.step_size, 1e-12),
                    "movement_ratio_over_n_steps_step_size": stats["movement_ratio"] / max(case.n_steps * case.step_size, 1e-12),
                    **stats,
                }
            )
        except Exception as exc:  # pragma: no cover - env dependent
            row.update({"status": "failure", "error": str(exc)})
        rows.append(row)

    successful = [r for r in rows if r.get("status") == "success"]
    if successful:
        best_endpoint = min(successful, key=lambda r: float(r.get("endpoint_error_mean", 1e99)))
        best_movement = min(successful, key=lambda r: abs(float(r.get("movement_ratio", 0.0)) - 1.0))
        # Simple scale fit: mean distance_ratio / step_size across successful cases.
        scale_distance_over_step = [float(r["distance_ratio_over_step_size"]) for r in successful if r.get("step_size", 0) != 0]
        scale_movement_over_step = [float(r["movement_ratio_over_step_size"]) for r in successful if r.get("step_size", 0) != 0]
        summary = {
            "scale_probe_case_count": len(rows),
            "scale_probe_success_count": len(successful),
            "scale_probe_failure_count": len(rows) - len(successful),
            "scale_probe_best_endpoint_error": float(best_endpoint.get("endpoint_error_mean", 0.0)),
            "scale_probe_best_endpoint_case": best_endpoint.get("case_id"),
            "scale_probe_best_endpoint_step_size": float(best_endpoint.get("step_size", 0.0)),
            "scale_probe_best_endpoint_n_steps": int(best_endpoint.get("n_steps", 0)),
            "scale_probe_best_movement_ratio": float(best_movement.get("movement_ratio", 0.0)),
            "scale_probe_best_movement_case": best_movement.get("case_id"),
            "scale_probe_distance_ratio_min": float(min(float(r.get("distance_ratio", 0.0)) for r in successful)),
            "scale_probe_distance_ratio_max": float(max(float(r.get("distance_ratio", 0.0)) for r in successful)),
            "scale_probe_movement_ratio_min": float(min(float(r.get("movement_ratio", 0.0)) for r in successful)),
            "scale_probe_movement_ratio_max": float(max(float(r.get("movement_ratio", 0.0)) for r in successful)),
            "scale_probe_distance_ratio_over_step_size_mean": float(sum(scale_distance_over_step) / len(scale_distance_over_step)),
            "scale_probe_movement_ratio_over_step_size_mean": float(sum(scale_movement_over_step) / len(scale_movement_over_step)),
        }
        # Human-readable verdict.
        if abs(summary["scale_probe_distance_ratio_over_step_size_mean"] - 1.0) < 0.1:
            distance_verdict = "distance_scales_like_step_size"
        elif abs(summary["scale_probe_distance_ratio_over_step_size_mean"] - 100.0) < 10.0:
            distance_verdict = "distance_inverse_step_size_suspected"
        else:
            distance_verdict = "distance_scaling_unclear"
        if summary["scale_probe_best_endpoint_error"] <= 1e-3:
            endpoint_verdict = "some_setting_endpoint_valid"
        else:
            endpoint_verdict = "no_setting_endpoint_valid"
        summary["scale_probe_verdict"] = f"{endpoint_verdict};{distance_verdict}"
    else:
        summary = {
            "scale_probe_case_count": len(rows),
            "scale_probe_success_count": 0,
            "scale_probe_failure_count": len(rows),
            "scale_probe_verdict": "all_calls_failed",
        }

    lrw_info.update({"metric_class": metric.__class__.__name__})
    return rows, lrw_info, summary
