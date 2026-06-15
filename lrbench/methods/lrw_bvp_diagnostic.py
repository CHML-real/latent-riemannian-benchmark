from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from lrbench.methods.lrw_bvp_solver import lrw_bvp_path_metrics, normalize_lrw_path


def clamp_path_endpoints(path: torch.Tensor, z0_batch: torch.Tensor, z1_batch: torch.Tensor) -> torch.Tensor:
    """Return a copy of path with the first/last points forced to z0/z1."""
    clamped = path.clone()
    clamped[0] = z0_batch
    clamped[-1] = z1_batch
    return clamped


def extract_solver_output(output: Any) -> tuple[Any, dict[str, Any]]:
    """Normalize LRW solver outputs into (raw_path, info)."""
    if isinstance(output, tuple) and len(output) >= 2:
        raw_path, solver_info = output[0], output[1]
    else:
        raw_path, solver_info = output, {}
    if not isinstance(solver_info, dict):
        solver_info = {"raw_solver_info": str(solver_info)}
    return raw_path, solver_info


def tensor_preview(x: torch.Tensor, max_rows: int = 3) -> list:
    """Small JSON-safe preview for diagnostics."""
    if not isinstance(x, torch.Tensor):
        return []
    with torch.no_grad():
        y = x.detach().cpu()
        if y.ndim == 0:
            return [float(y.item())]
        if y.ndim == 1:
            return y[:max_rows].tolist()
        return y.reshape(-1, y.shape[-1])[:max_rows].tolist()


def summarize_bvp_path(
    path: torch.Tensor,
    z0_batch: torch.Tensor,
    z1_batch: torch.Tensor,
    endpoint_tolerance: float,
    beta: float = 6.0,
    sigma: float = 0.7,
    cross: float = 0.2,
    regularization: float = 1e-5,
) -> dict[str, Any]:
    """Raw + endpoint-clamped diagnostics for a normalized path."""
    raw_metrics = lrw_bvp_path_metrics(path, z0_batch, z1_batch, beta=beta, sigma=sigma, cross=cross, regularization=regularization)
    clamped = clamp_path_endpoints(path, z0_batch, z1_batch)
    clamped_metrics = lrw_bvp_path_metrics(clamped, z0_batch, z1_batch, beta=beta, sigma=sigma, cross=cross, regularization=regularization)

    raw_endpoint = float(raw_metrics.get("lrw_bvp_endpoint_error_mean", float("inf")))
    clamped_endpoint = float(clamped_metrics.get("lrw_bvp_endpoint_error_mean", float("inf")))

    return {
        "raw": raw_metrics,
        "clamped": {f"clamped_{k}": v for k, v in clamped_metrics.items()},
        "endpoint_tolerance": float(endpoint_tolerance),
        "raw_endpoint_pass": raw_endpoint <= endpoint_tolerance,
        "clamped_endpoint_pass": clamped_endpoint <= endpoint_tolerance,
        "raw_start_preview": tensor_preview(path[0]),
        "raw_end_preview": tensor_preview(path[-1]),
        "target_start_preview": tensor_preview(z0_batch),
        "target_end_preview": tensor_preview(z1_batch),
    }


def _to_float(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _safe_ratio(value: float | None, denom: float) -> float | None:
    if value is None:
        return None
    return value / max(float(denom), 1e-12)


def case_validity_flags(case_result: dict[str, Any], endpoint_tolerance: float | None = None, energy_ratio_threshold: float = 1.0) -> dict[str, Any]:
    """Separate call success, endpoint validity, energy improvement, and overall validity."""
    if case_result.get("status") != "success":
        return {
            "call_success": False,
            "endpoint_valid": False,
            "energy_improved": False,
            "overall_valid": False,
            "valid_geodesic_score": 0.0,
        }

    diagnostics = case_result.get("diagnostics", {})
    raw = diagnostics.get("raw", {})
    endpoint_error = _to_float(raw.get("lrw_bvp_endpoint_error_mean"), float("inf"))
    energy_ratio = _to_float(raw.get("lrw_bvp_energy_ratio_over_straight"), None)
    if endpoint_tolerance is None:
        endpoint_tolerance = _to_float(diagnostics.get("endpoint_tolerance"), 1e-3) or 1e-3

    endpoint_valid = bool(endpoint_error is not None and endpoint_error <= endpoint_tolerance)
    energy_improved = bool(energy_ratio is not None and energy_ratio <= energy_ratio_threshold)
    overall_valid = endpoint_valid and energy_improved

    # A bounded score that cannot hide endpoint failure. It is useful only for ranking candidates.
    endpoint_score = 1.0 / (1.0 + max(float(endpoint_error or 0.0), 0.0) / max(endpoint_tolerance, 1e-12))
    energy_score = 0.0 if energy_ratio is None else max(0.0, min(1.0, 1.0 - float(energy_ratio)))
    valid_geodesic_score = endpoint_score * (0.25 + 0.75 * energy_score)
    if overall_valid:
        valid_geodesic_score += 1.0

    return {
        "call_success": True,
        "endpoint_valid": endpoint_valid,
        "energy_improved": energy_improved,
        "overall_valid": overall_valid,
        "valid_geodesic_score": float(valid_geodesic_score),
        "endpoint_error_mean": endpoint_error,
        "energy_ratio_over_straight": energy_ratio,
    }


def compact_case_metrics(case_result: dict[str, Any]) -> dict[str, Any]:
    """Flatten the most important case-level diagnostic metrics."""
    if case_result.get("status") != "success":
        flags = case_validity_flags(case_result)
        return {
            "status": case_result.get("status"),
            "failure_type": case_result.get("failure", {}).get("failure_type"),
            **flags,
        }
    raw = case_result.get("diagnostics", {}).get("raw", {})
    clamped = case_result.get("diagnostics", {}).get("clamped", {})
    flags = case_validity_flags(case_result)
    return {
        "status": case_result.get("status"),
        "raw_endpoint_error_mean": raw.get("lrw_bvp_endpoint_error_mean"),
        "raw_energy_mean": raw.get("lrw_bvp_pullback_energy_mean"),
        "raw_length_mean": raw.get("lrw_bvp_pullback_length_mean"),
        "raw_energy_ratio_over_straight": raw.get("lrw_bvp_energy_ratio_over_straight"),
        "clamped_endpoint_error_mean": clamped.get("clamped_lrw_bvp_endpoint_error_mean"),
        "clamped_energy_mean": clamped.get("clamped_lrw_bvp_pullback_energy_mean"),
        "clamped_length_mean": clamped.get("clamped_lrw_bvp_pullback_length_mean"),
        "clamped_energy_ratio_over_straight": clamped.get("clamped_lrw_bvp_energy_ratio_over_straight"),
        "raw_endpoint_pass": case_result.get("diagnostics", {}).get("raw_endpoint_pass"),
        "clamped_endpoint_pass": case_result.get("diagnostics", {}).get("clamped_endpoint_pass"),
        **flags,
    }


def _case_sort_value(case: dict[str, Any], key_path: str, default: float = float("inf")) -> float:
    cur: Any = case
    for key in key_path.split("."):
        cur = cur.get(key, {}) if isinstance(cur, dict) else {}
    value = _to_float(cur, default)
    return float(default if value is None else value)


def _case_summary(case: dict[str, Any] | None) -> dict[str, Any] | None:
    if case is None:
        return None
    compact = case.get("compact", {})
    return {
        "case_id": case.get("case_id"),
        "method_name": case.get("method_name"),
        "status": case.get("status"),
        "params": case.get("params", {}),
        "raw_endpoint_error_mean": compact.get("raw_endpoint_error_mean"),
        "raw_energy_mean": compact.get("raw_energy_mean"),
        "raw_length_mean": compact.get("raw_length_mean"),
        "raw_energy_ratio_over_straight": compact.get("raw_energy_ratio_over_straight"),
        "clamped_endpoint_error_mean": compact.get("clamped_endpoint_error_mean"),
        "clamped_energy_mean": compact.get("clamped_energy_mean"),
        "valid_geodesic_score": compact.get("valid_geodesic_score"),
        "endpoint_valid": compact.get("endpoint_valid"),
        "energy_improved": compact.get("energy_improved"),
        "overall_valid": compact.get("overall_valid"),
        "failure_type": case.get("failure", {}).get("failure_type"),
    }


def select_best_cases(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick best cases by endpoint, energy, and validity-aware score."""
    successes = [c for c in case_results if c.get("status") == "success"]
    if not successes:
        return {
            "best_by_endpoint": None,
            "best_by_energy": None,
            "best_by_valid_score": None,
            "best_overall_valid": None,
        }
    best_by_endpoint = min(successes, key=lambda c: _case_sort_value(c, "compact.raw_endpoint_error_mean"))
    best_by_energy = min(successes, key=lambda c: _case_sort_value(c, "compact.raw_energy_ratio_over_straight"))
    best_by_valid_score = max(successes, key=lambda c: _case_sort_value(c, "compact.valid_geodesic_score", default=0.0))
    valid_cases = [c for c in successes if c.get("compact", {}).get("overall_valid")]
    best_overall_valid = max(valid_cases, key=lambda c: _case_sort_value(c, "compact.valid_geodesic_score", default=0.0)) if valid_cases else None
    return {
        "best_by_endpoint": _case_summary(best_by_endpoint),
        "best_by_energy": _case_summary(best_by_energy),
        "best_by_valid_score": _case_summary(best_by_valid_score),
        "best_overall_valid": _case_summary(best_overall_valid),
    }


def aggregate_diagnostic_cases(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate diagnostic cases without turning endpoint miss into runner failure."""
    total = len(case_results)
    call_success = [c for c in case_results if c.get("status") == "success"]
    raw_endpoint_pass = [c for c in call_success if c.get("diagnostics", {}).get("raw_endpoint_pass")]
    clamped_endpoint_pass = [c for c in call_success if c.get("diagnostics", {}).get("clamped_endpoint_pass")]
    energy_improved = [c for c in call_success if c.get("compact", {}).get("energy_improved")]
    overall_valid = [c for c in call_success if c.get("compact", {}).get("overall_valid")]
    endpoint_fail_energy_improved = [c for c in energy_improved if not c.get("compact", {}).get("endpoint_valid")]

    def _values(path: str) -> list[float]:
        keys = path.split(".")
        values: list[float] = []
        for c in call_success:
            cur: Any = c
            for key in keys:
                cur = cur.get(key, {}) if isinstance(cur, dict) else {}
            if isinstance(cur, (int, float)):
                values.append(float(cur))
        return values

    raw_endpoint_errors = _values("diagnostics.raw.lrw_bvp_endpoint_error_mean")
    raw_energy = _values("diagnostics.raw.lrw_bvp_pullback_energy_mean")
    clamped_energy = _values("diagnostics.clamped.clamped_lrw_bvp_pullback_energy_mean")
    raw_energy_ratios = _values("diagnostics.raw.lrw_bvp_energy_ratio_over_straight")
    best_cases = select_best_cases(case_results)

    return {
        "case_count": total,
        "call_success_count": len(call_success),
        "call_failure_count": total - len(call_success),
        "call_failure_rate": 0.0 if total == 0 else (total - len(call_success)) / total,
        "raw_endpoint_pass_count": len(raw_endpoint_pass),
        "raw_endpoint_failure_count": len(call_success) - len(raw_endpoint_pass),
        "raw_endpoint_failure_rate": 0.0 if len(call_success) == 0 else (len(call_success) - len(raw_endpoint_pass)) / len(call_success),
        "clamped_endpoint_pass_count": len(clamped_endpoint_pass),
        "energy_improved_count": len(energy_improved),
        "energy_only_success_count": len(endpoint_fail_energy_improved),
        "overall_valid_count": len(overall_valid),
        "overall_valid_rate": 0.0 if len(call_success) == 0 else len(overall_valid) / len(call_success),
        "endpoint_tolerance_pass_rate": 0.0 if len(call_success) == 0 else len(raw_endpoint_pass) / len(call_success),
        "boundary_condition_failure_count": len(call_success) - len(raw_endpoint_pass),
        "raw_endpoint_error_mean_min": min(raw_endpoint_errors) if raw_endpoint_errors else None,
        "raw_endpoint_error_mean_max": max(raw_endpoint_errors) if raw_endpoint_errors else None,
        "raw_energy_mean_min": min(raw_energy) if raw_energy else None,
        "raw_energy_mean_max": max(raw_energy) if raw_energy else None,
        "raw_energy_ratio_min": min(raw_energy_ratios) if raw_energy_ratios else None,
        "raw_energy_ratio_max": max(raw_energy_ratios) if raw_energy_ratios else None,
        "clamped_energy_mean_min": min(clamped_energy) if clamped_energy else None,
        "clamped_energy_mean_max": max(clamped_energy) if clamped_energy else None,
        **best_cases,
    }
