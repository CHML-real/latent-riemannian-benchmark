from __future__ import annotations

import hashlib
import importlib
import inspect
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SourceTarget:
    target_id: str
    import_path: str
    attr_chain: tuple[str, ...]


def default_source_targets() -> list[SourceTarget]:
    return [
        SourceTarget("geodesic_solver_class", "lrw.geodesic.solver", ("GeodesicSolver",)),
        SourceTarget("geodesic_solver_interpolate", "lrw.geodesic.solver", ("GeodesicSolver", "interpolate")),
        SourceTarget("geodesic_solver_geodesic_distance", "lrw.geodesic.solver", ("GeodesicSolver", "geodesic_distance")),
        SourceTarget("geodesic_solver_shoot", "lrw.geodesic.solver", ("GeodesicSolver", "shoot")),
        SourceTarget("bvp_solver_class", "lrw.geodesic.bvp", ("BVPSolver",)),
        SourceTarget("bvp_solver_geodesic_path", "lrw.geodesic.bvp", ("BVPSolver", "geodesic_path")),
        SourceTarget("bvp_solver_solve", "lrw.geodesic.bvp", ("BVPSolver", "solve")),
        SourceTarget("pullback_metric_class", "lrw.metric.pullback", ("PullbackMetric",)),
        SourceTarget("pullback_metric_metric_tensor", "lrw.metric.pullback", ("PullbackMetric", "metric_tensor")),
        SourceTarget("slerp_function", "lrw.geodesic", ("slerp",)),
        SourceTarget("slerp_path_function", "lrw.geodesic", ("slerp_path",)),
    ]


def resolve_object(module_name: str, attr_chain: tuple[str, ...]) -> Any:
    module = importlib.import_module(module_name)
    obj: Any = module
    for attr in attr_chain:
        obj = getattr(obj, attr)
    return obj


def _safe_signature(obj: Any) -> str | None:
    try:
        return str(inspect.signature(obj))
    except Exception:
        return None


def _source_lines(obj: Any) -> tuple[list[str], int | None, str | None]:
    try:
        lines, start_line = inspect.getsourcelines(obj)
        return [line.rstrip("\n") for line in lines], int(start_line), inspect.getsourcefile(obj)
    except Exception:
        try:
            source = inspect.getsource(obj)
            return source.splitlines(), None, inspect.getsourcefile(obj)
        except Exception:
            return [], None, None


def _snippet(lines: list[str], max_lines: int) -> list[str]:
    if max_lines <= 0:
        return []
    return lines[:max_lines]


def _contains_flags(text: str) -> dict[str, bool]:
    needles = {
        "contains_step_size": "step_size",
        "contains_n_steps": "n_steps",
        "contains_shoot": "shoot",
        "contains_interpolate": "interpolate",
        "contains_geodesic_distance": "geodesic_distance",
        "contains_solve": "solve",
        "contains_num_points": "num_points",
        "contains_linspace": "linspace",
        "contains_stack": "stack",
        "contains_cat": "cat",
        "contains_metric_tensor": "metric_tensor",
        "contains_christoffel": "christoffel",
        "contains_geodesic_acceleration": "geodesic_acceleration",
        "contains_final_error": "final_error",
        "contains_converged": "converged",
        "contains_torch_norm": "norm",
    }
    return {key: (needle in text) for key, needle in needles.items()}


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def probe_source_target(target: SourceTarget, max_source_lines: int = 80) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "target_id": target.target_id,
        "import_path": target.import_path,
        "attr_chain": ".".join(target.attr_chain),
        "qualified_path": target.import_path + ":" + ".".join(target.attr_chain),
    }
    try:
        obj = resolve_object(target.import_path, target.attr_chain)
    except Exception as exc:
        entry.update({"status": "failure", "error": str(exc)})
        return entry

    lines, start_line, source_file = _source_lines(obj)
    text = "\n".join(lines)
    doc = inspect.getdoc(obj) or ""
    sig = _safe_signature(obj)
    flags = _contains_flags(text)
    entry.update(
        {
            "status": "success",
            "object_type": type(obj).__name__,
            "module": getattr(obj, "__module__", None),
            "qualname": getattr(obj, "__qualname__", None),
            "signature": sig,
            "docstring_present": bool(doc),
            "docstring_first_line": doc.splitlines()[0] if doc.splitlines() else "",
            "source_file": source_file,
            "source_start_line": start_line,
            "source_line_count": len(lines),
            "source_sha256_16": _hash_text(text),
            "source_snippet": _snippet(lines, max_source_lines),
            **flags,
        }
    )
    return entry


def run_lrw_source_probe(max_source_lines: int = 80) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    try:
        lrw = importlib.import_module("lrw")
        lrw_available = True
        reason = None
        version = getattr(lrw, "__version__", None)
        module_file = getattr(lrw, "__file__", None)
    except Exception as exc:
        return [], {"available": False, "reason": str(exc)}, {
            "source_probe_target_count": 0,
            "source_probe_success_count": 0,
            "source_probe_failure_count": 0,
            "source_probe_verdict": "lrw_unavailable",
        }

    entries = [probe_source_target(t, max_source_lines=max_source_lines) for t in default_source_targets()]
    successes = [e for e in entries if e.get("status") == "success"]
    failures = [e for e in entries if e.get("status") != "success"]

    def by_id(tid: str) -> dict[str, Any]:
        return next((e for e in entries if e.get("target_id") == tid), {})

    gd = by_id("geodesic_solver_geodesic_distance")
    interp = by_id("geodesic_solver_interpolate")
    bvp_path = by_id("bvp_solver_geodesic_path")
    bvp_solve = by_id("bvp_solver_solve")

    step_size_refs = sum(1 for e in successes if e.get("contains_step_size"))
    shoot_refs = sum(1 for e in successes if e.get("contains_shoot"))
    num_points_refs = sum(1 for e in successes if e.get("contains_num_points"))

    verdict_bits: list[str] = []
    if gd.get("contains_step_size"):
        verdict_bits.append("geodesic_distance_source_mentions_step_size")
    if interp.get("contains_shoot"):
        verdict_bits.append("interpolate_source_mentions_shoot")
    if interp.get("contains_step_size"):
        verdict_bits.append("interpolate_source_mentions_step_size")
    if bvp_path.get("contains_solve"):
        verdict_bits.append("bvp_geodesic_path_source_mentions_solve")
    if bvp_solve.get("contains_num_points"):
        verdict_bits.append("bvp_solve_source_mentions_num_points")
    if not verdict_bits:
        verdict_bits.append("no_key_source_keywords_detected")

    summary = {
        "source_probe_target_count": len(entries),
        "source_probe_success_count": len(successes),
        "source_probe_failure_count": len(failures),
        "source_probe_step_size_reference_count": step_size_refs,
        "source_probe_shoot_reference_count": shoot_refs,
        "source_probe_num_points_reference_count": num_points_refs,
        "source_probe_geodesic_distance_contains_step_size": bool(gd.get("contains_step_size", False)),
        "source_probe_interpolate_contains_shoot": bool(interp.get("contains_shoot", False)),
        "source_probe_interpolate_contains_step_size": bool(interp.get("contains_step_size", False)),
        "source_probe_bvp_path_contains_solve": bool(bvp_path.get("contains_solve", False)),
        "source_probe_bvp_solve_contains_num_points": bool(bvp_solve.get("contains_num_points", False)),
        "source_probe_verdict": ";".join(verdict_bits),
    }
    lrw_info = {"available": lrw_available, "reason": reason, "version": version, "module_file": module_file}
    return entries, lrw_info, summary
