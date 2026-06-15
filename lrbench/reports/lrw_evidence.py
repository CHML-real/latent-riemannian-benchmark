from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_results(raw_dir: str | Path) -> list[dict[str, Any]]:
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"raw result directory does not exist: {raw_path}")
    results: list[dict[str, Any]] = []
    for path in sorted(raw_path.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("benchmark_id"):
            data["_source_file"] = str(path)
            results.append(data)
    return results


def metrics_of(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    summary = result.get("summary")
    if isinstance(summary, dict):
        return summary
    return {}


def by_id(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r.get("benchmark_id")): r for r in results}


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 1e-4 or abs(value) >= 1e5:
            return f"{value:.3e}"
        return f"{value:.6g}"
    return str(value)


@dataclass(frozen=True)
class EvidenceRow:
    layer: str
    component: str
    benchmark_id: str
    status: str
    key_evidence: str
    interpretation: str
    failure_mode: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "layer": self.layer,
            "component": self.component,
            "benchmark_id": self.benchmark_id,
            "status": self.status,
            "key_evidence": self.key_evidence,
            "interpretation": self.interpretation,
            "failure_mode": self.failure_mode,
        }


def _status(result: dict[str, Any] | None) -> str:
    if not result:
        return "missing"
    return str(result.get("status", "unknown"))


def build_evidence_rows(results: list[dict[str, Any]]) -> list[EvidenceRow]:
    idx = by_id(results)
    rows: list[EvidenceRow] = []

    r = idx.get("lrw_probe_001")
    m = metrics_of(r)
    rows.append(EvidenceRow(
        "Adapter availability",
        "LRW package import",
        "lrw_probe_001",
        _status(r),
        f"available={fmt(m.get('available'))}; public_symbols={fmt(m.get('public_symbol_count'))}; submodules={fmt(m.get('candidate_submodule_count'))}",
        "The installed LRW package is importable and exposes the expected top-level modules.",
    ))

    r = idx.get("lrw_api_introspection_001")
    m = metrics_of(r)
    rows.append(EvidenceRow(
        "API surface",
        "LRW geodesic/metric classes",
        "lrw_api_introspection_001",
        _status(r),
        f"classes={fmt(m.get('api_importable_class_count'))}/{fmt(m.get('api_class_count'))}; methods={fmt(m.get('api_method_count'))}; signatures={fmt(m.get('api_signature_count'))}",
        "The expected solver, BVP, pullback metric, and base metric APIs are visible for benchmarking.",
    ))

    r = idx.get("lrw_pullback_metric_001")
    m = metrics_of(r)
    rows.append(EvidenceRow(
        "Metric layer",
        "PullbackMetric.metric_tensor",
        "lrw_pullback_metric_001",
        _status(r),
        f"spd_violation={fmt(m.get('lrw_pullback_spd_violation_rate'))}; ref_rel_frob={fmt(m.get('lrw_vs_reference_relative_frobenius_error_mean'))}; max_abs={fmt(m.get('lrw_vs_reference_max_abs_error'))}",
        "LRW PullbackMetric matches the reference toy-decoder pullback metric and is reliable in this benchmark.",
    ))

    r = idx.get("lrw_slerp_001")
    m = metrics_of(r)
    rows.append(EvidenceRow(
        "Path function layer",
        "slerp_path",
        "lrw_slerp_001",
        _status(r),
        f"endpoint={fmt(m.get('lrw_slerp_endpoint_error_mean'))}; sphere_dist_err={fmt(m.get('lrw_slerp_sphere_distance_error_mean'))}; radial_max={fmt(m.get('lrw_slerp_sphere_radial_error_max'))}; ref_rel={fmt(m.get('lrw_slerp_vs_reference_relative_error_mean'))}",
        "The simple spherical interpolation path function preserves endpoints and sphere geometry.",
    ))

    r = idx.get("lrw_geodesic_solver_euclidean_001")
    m = metrics_of(r)
    rows.append(EvidenceRow(
        "Solver layer",
        "GeodesicSolver.interpolate / geodesic_distance",
        "lrw_geodesic_solver_euclidean_001",
        _status(r),
        f"endpoint={fmt(m.get('lrw_geodesic_solver_endpoint_error_mean'))}; path_dist_err={fmt(m.get('lrw_geodesic_solver_distance_error_mean'))}; geodesic_dist_err={fmt(m.get('lrw_geodesic_solver_geodesic_distance_error_mean'))}; energy_ratio={fmt(m.get('lrw_geodesic_solver_energy_ratio_over_straight'))}",
        "In flat Euclidean sanity testing, GeodesicSolver does not behave as an endpoint-preserving interpolation/distance oracle.",
        "GEO_F01_ENDPOINT_MISS; DIST_F01_STEP_SIZE_SCALED",
    ))

    r = idx.get("lrw_euclidean_bvp_001")
    m = metrics_of(r)
    rows.append(EvidenceRow(
        "BVP solver layer",
        "BVPSolver.geodesic_path on Euclidean metric",
        "lrw_euclidean_bvp_001",
        _status(r),
        f"endpoint={fmt(m.get('lrw_euclidean_bvp_endpoint_error_mean'))}; distance_err={fmt(m.get('lrw_euclidean_bvp_distance_error_mean'))}; energy_ratio={fmt(m.get('lrw_euclidean_bvp_energy_ratio_over_straight'))}",
        "The BVP solver misses endpoint validity even in the flat Euclidean sanity case.",
        "GEO_F01_ENDPOINT_MISS",
    ))

    r = idx.get("lrw_bvp_001")
    m = metrics_of(r)
    rows.append(EvidenceRow(
        "BVP solver layer",
        "BVPSolver.geodesic_path on toy pullback metric",
        "lrw_bvp_001",
        _status(r),
        f"endpoint={fmt(m.get('lrw_bvp_endpoint_error_mean'))}; energy_ratio={fmt(m.get('lrw_bvp_energy_ratio_over_straight'))}; returned_points={fmt(m.get('lrw_bvp_num_points_returned'))}",
        "The pullback BVP can reduce energy but cannot be accepted as a valid BVP geodesic when endpoints miss.",
        "GEO_F01_ENDPOINT_MISS; ENERGY_ONLY_SUCCESS",
    ))

    r = idx.get("lrw_bvp_diagnostic_001")
    m = metrics_of(r)
    rows.append(EvidenceRow(
        "BVP diagnostics",
        "BVP best-case and validity scoring",
        "lrw_bvp_diagnostic_001",
        _status(r),
        f"case_count={fmt(m.get('diagnostic_case_count'))}; raw_endpoint_pass={fmt(m.get('diagnostic_raw_endpoint_pass_count'))}; energy_only={fmt(m.get('diagnostic_energy_only_success_count'))}; overall_valid={fmt(m.get('diagnostic_overall_valid_count'))}; best_endpoint={fmt(m.get('diagnostic_best_raw_endpoint_error'))}",
        "Energy improvement and endpoint validity are separated; no tested BVP case is overall valid.",
        "ENERGY_ONLY_SUCCESS; BVP_OVERALL_INVALID",
    ))

    r = idx.get("lrw_geodesic_output_anatomy_001")
    m = metrics_of(r)
    rows.append(EvidenceRow(
        "Output anatomy",
        "Solver output shape semantics",
        "lrw_geodesic_output_anatomy_001",
        _status(r),
        f"axis_candidates={fmt(m.get('anatomy_axis_candidate_count'))}; endpoint_valid_axes={fmt(m.get('anatomy_axis_endpoint_valid_count'))}; best_axis_endpoint={fmt(m.get('anatomy_best_axis_endpoint_error'))}; verdict={fmt(m.get('anatomy_verdict'))}",
        "The endpoint miss is not explained by a simple [T,B,D] vs [B,T,D] axis interpretation error.",
        "API_SEM02_NO_AXIS_INTERPRETATION_PASS",
    ))

    r = idx.get("lrw_solver_scale_probe_001")
    m = metrics_of(r)
    rows.append(EvidenceRow(
        "Scale semantics",
        "step_size and n_steps probe",
        "lrw_solver_scale_probe_001",
        _status(r),
        f"cases={fmt(m.get('scale_probe_case_count'))}; best_endpoint={fmt(m.get('scale_probe_best_endpoint_error'))}; dist_ratio_range={fmt(m.get('scale_probe_distance_ratio_min'))}..{fmt(m.get('scale_probe_distance_ratio_max'))}; dist_ratio_over_step={fmt(m.get('scale_probe_distance_ratio_over_step_size_mean'))}; verdict={fmt(m.get('scale_probe_verdict'))}",
        "geodesic_distance scales like step_size; no tested step_size/n_steps setting makes interpolate endpoint-valid.",
        "DIST_F01_STEP_SIZE_SCALED; GEO_F01_ENDPOINT_MISS",
    ))

    r = idx.get("lrw_source_probe_001")
    m = metrics_of(r)
    rows.append(EvidenceRow(
        "Implementation fingerprint",
        "LRW source/docstring probe",
        "lrw_source_probe_001",
        _status(r),
        f"targets={fmt(m.get('source_probe_target_count'))}; step_refs={fmt(m.get('source_probe_step_size_reference_count'))}; shoot_refs={fmt(m.get('source_probe_shoot_reference_count'))}; gd_step={fmt(m.get('source_probe_geodesic_distance_contains_step_size'))}; interp_shoot={fmt(m.get('source_probe_interpolate_contains_shoot'))}",
        "The installed source fingerprint supports the experimental reading: distance and interpolate are step/shoot-based rather than plain endpoint interpolation APIs.",
        "API_SEM01_INTERPOLATE_IS_SHOOT_BASED; DIST_F01_STEP_SIZE_SCALED",
    ))

    return rows


def build_summary_metrics(rows: list[EvidenceRow]) -> dict[str, Any]:
    pass_count = sum(1 for r in rows if r.status == "success")
    fail_count = sum(1 for r in rows if r.status == "failure")
    missing_count = sum(1 for r in rows if r.status == "missing")
    trusted_components = [r.component for r in rows if r.status == "success" and r.layer in {"Metric layer", "Path function layer"}]
    risky_components = [r.component for r in rows if r.failure_mode]
    return {
        "evidence_row_count": len(rows),
        "evidence_pass_count": pass_count,
        "evidence_fail_count": fail_count,
        "evidence_missing_count": missing_count,
        "trusted_component_count": len(trusted_components),
        "risky_component_count": len(risky_components),
    }


def write_evidence_csv(rows: list[EvidenceRow], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["layer", "component", "benchmark_id", "status", "key_evidence", "interpretation", "failure_mode"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def make_evidence_markdown(results: list[dict[str, Any]], rows: list[EvidenceRow]) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    metrics = build_summary_metrics(rows)
    lines: list[str] = []
    lines.append("# LRW Adapter Evidence Report")
    lines.append("")
    lines.append(f"Generated at: `{generated_at}`")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("This report aggregates the LRW adapter benchmark results into a component-level evidence bundle.")
    lines.append("")
    lines.append("- **Reliable in this benchmark:** `PullbackMetric.metric_tensor`, `slerp_path`, package import/API surface.")
    lines.append("- **Risky as endpoint geodesic APIs in this benchmark:** `GeodesicSolver.interpolate`, `GeodesicSolver.geodesic_distance`, `BVPSolver.geodesic_path`.")
    lines.append("- **Main diagnosis:** LRW metric/path-function layers pass, but solver-layer endpoint and distance semantics do not match the benchmark's endpoint-preserving geodesic expectations.")
    lines.append("- **Key semantic evidence:** `geodesic_distance` scales like `step_size`; `interpolate` source references `shoot` and `step_size`; no tested axis interpretation makes solver outputs endpoint-valid.")
    lines.append("")
    lines.append("## Evidence Counts")
    lines.append("")
    for key, value in metrics.items():
        lines.append(f"- {key}: `{fmt(value)}`")
    lines.append("")
    lines.append("## Evidence Matrix")
    lines.append("")
    lines.append("| Layer | Component | Benchmark | Status | Key evidence | Interpretation | Failure mode |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row.layer} | `{row.component}` | `{row.benchmark_id}` | `{row.status}` | {row.key_evidence} | {row.interpretation} | `{row.failure_mode}` |"
        )
    lines.append("")
    lines.append("## Layered Diagnosis")
    lines.append("")
    lines.append("### Metric layer")
    lines.append("")
    lines.append("`PullbackMetric.metric_tensor` matches the reference toy decoder pullback metric with zero relative Frobenius error in the benchmark output. This supports using LRW's metric tensor implementation for metric evaluation.")
    lines.append("")
    lines.append("### Simple path-function layer")
    lines.append("")
    lines.append("`slerp_path` passes endpoint, radial, sphere-distance, and reference-comparison checks. This suggests that LRW's simple spherical interpolation path function is reliable under the tested conditions.")
    lines.append("")
    lines.append("### Solver layer")
    lines.append("")
    lines.append("`GeodesicSolver.interpolate`, `GeodesicSolver.geodesic_distance`, and `BVPSolver.geodesic_path` fail endpoint/distance sanity checks under the benchmark's geodesic API expectations. The failure is not explained by simple tensor-axis misinterpretation, and the scale probe shows distance values scale like `step_size`.")
    lines.append("")
    lines.append("## Failure Modes")
    lines.append("")
    lines.append("| Code | Meaning | Evidence |")
    lines.append("|---|---|---|")
    lines.append("| `GEO_F01_ENDPOINT_MISS` | Returned path does not satisfy endpoint boundary condition. | Pullback BVP, Euclidean BVP, GeodesicSolver Euclidean, anatomy, and scale probe. |")
    lines.append("| `DIST_F01_STEP_SIZE_SCALED` | Distance output scales with solver `step_size` instead of directly matching Euclidean distance. | Scale probe and source probe. |")
    lines.append("| `API_SEM01_INTERPOLATE_IS_SHOOT_BASED` | `interpolate` appears to be based on shooting/step rollout semantics rather than guaranteed endpoint interpolation. | Source probe flags and movement-ratio behavior. |")
    lines.append("| `API_SEM02_NO_AXIS_INTERPRETATION_PASS` | Endpoint miss is not fixed by common tensor-axis interpretations. | Output anatomy probe. |")
    lines.append("| `ENERGY_ONLY_SUCCESS` | Energy can improve while endpoint validity still fails. | BVP diagnostic best-case scoring. |")
    lines.append("")
    lines.append("## Repro Commands")
    lines.append("")
    lines.append("```bat")
    lines.append("python -m lrbench.runners.run_lrw_probe --config configs/lrw_probe.yaml")
    lines.append("python -m lrbench.runners.run_lrw_introspection --config configs/lrw_api_introspection.yaml")
    lines.append("python -m lrbench.runners.run_lrw_pullback_metric --config configs/lrw_pullback_metric.yaml")
    lines.append("python -m lrbench.runners.run_lrw_bvp --config configs/lrw_bvp.yaml")
    lines.append("python -m lrbench.runners.run_lrw_bvp_diagnostic --config configs/lrw_bvp_diagnostic.yaml")
    lines.append("python -m lrbench.runners.run_lrw_euclidean_bvp --config configs/lrw_euclidean_bvp.yaml")
    lines.append("python -m lrbench.runners.run_lrw_geodesic_solver --config configs/lrw_geodesic_solver.yaml")
    lines.append("python -m lrbench.runners.run_lrw_slerp --config configs/lrw_slerp.yaml")
    lines.append("python -m lrbench.runners.run_lrw_geodesic_anatomy --config configs/lrw_geodesic_anatomy.yaml")
    lines.append("python -m lrbench.runners.run_lrw_solver_scale_probe --config configs/lrw_solver_scale_probe.yaml")
    lines.append("python -m lrbench.runners.run_lrw_source_probe --config configs/lrw_source_probe.yaml")
    lines.append("python -m lrbench.runners.run_lrw_evidence_report --config configs/lrw_evidence_report.yaml")
    lines.append("```")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `results/reports/lrw_evidence_report.md`")
    lines.append("- `results/reports/lrw_evidence_matrix.csv`")
    lines.append("")
    return "\n".join(lines)


def generate_lrw_evidence_report(raw_dir: str | Path, report_dir: str | Path) -> dict[str, Any]:
    results = load_results(raw_dir)
    rows = build_evidence_rows(results)
    report_path = Path(report_dir) / "lrw_evidence_report.md"
    csv_path = Path(report_dir) / "lrw_evidence_matrix.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(make_evidence_markdown(results, rows), encoding="utf-8")
    write_evidence_csv(rows, csv_path)
    return {
        "status": "success",
        "raw_dir": str(raw_dir),
        "report_dir": str(report_dir),
        "markdown_path": str(report_path),
        "csv_path": str(csv_path),
        "metrics": build_summary_metrics(rows),
        "row_count": len(rows),
    }
