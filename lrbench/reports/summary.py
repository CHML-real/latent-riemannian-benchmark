from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IMPORTANT_METRIC_KEYS = [
    "endpoint_error_mean",
    "distance_error_mean",
    "sphere_distance_error_mean",
    "sphere_radial_error_max",
    "energy_improvement",
    "length_improvement",
    "spd_violation_rate",
    "pullback_spd_violation_rate",
    "lrw_pullback_spd_violation_rate",
    "lrw_vs_reference_relative_frobenius_error_mean",
    "lrw_vs_reference_max_abs_error",
    "sample_count",
    "lrw_bvp_endpoint_error_mean",
    "lrw_bvp_pullback_energy_mean",
    "lrw_bvp_pullback_length_mean",
    "lrw_bvp_energy_ratio_over_straight",
    "lrw_bvp_num_points_returned",
    "lrw_euclidean_bvp_endpoint_error_mean",
    "lrw_euclidean_bvp_distance_error_mean",
    "lrw_euclidean_bvp_energy_ratio_over_straight",
    "lrw_euclidean_bvp_num_points_returned",
    "lrw_geodesic_solver_endpoint_error_mean",
    "lrw_geodesic_solver_distance_error_mean",
    "lrw_geodesic_solver_geodesic_distance_error_mean",
    "lrw_geodesic_solver_energy_ratio_over_straight",
    "lrw_geodesic_solver_num_points_returned",
    "lrw_slerp_endpoint_error_mean",
    "lrw_slerp_sphere_distance_error_mean",
    "lrw_slerp_sphere_radial_error_max",
    "lrw_slerp_vs_reference_relative_error_mean",
    "lrw_slerp_vs_reference_max_abs_error",
    "lrw_slerp_num_points_returned",
    "anatomy_call_count",
    "anatomy_successful_call_count",
    "anatomy_axis_candidate_count",
    "anatomy_axis_endpoint_valid_count",
    "anatomy_best_axis_endpoint_error",
    "anatomy_best_axis_distance_error",
    "anatomy_best_axis_energy_ratio",
    "anatomy_geodesic_distance_relative_error_mean",
    "scale_probe_case_count",
    "scale_probe_success_count",
    "scale_probe_failure_count",
    "scale_probe_best_endpoint_error",
    "scale_probe_best_movement_ratio",
    "scale_probe_distance_ratio_min",
    "scale_probe_distance_ratio_max",
    "scale_probe_movement_ratio_min",
    "scale_probe_movement_ratio_max",
    "scale_probe_distance_ratio_over_step_size_mean",
    "scale_probe_movement_ratio_over_step_size_mean",
    "source_probe_target_count",
    "source_probe_success_count",
    "source_probe_failure_count",
    "source_probe_step_size_reference_count",
    "source_probe_shoot_reference_count",
    "source_probe_num_points_reference_count",
    "source_probe_geodesic_distance_contains_step_size",
    "source_probe_interpolate_contains_shoot",
    "source_probe_interpolate_contains_step_size",
    "source_probe_bvp_path_contains_solve",
    "source_probe_bvp_solve_contains_num_points",
    "source_probe_target_count",
    "source_probe_success_count",
    "source_probe_failure_count",
    "source_probe_step_size_reference_count",
    "source_probe_shoot_reference_count",
    "source_probe_num_points_reference_count",
    "source_probe_geodesic_distance_contains_step_size",
    "source_probe_interpolate_contains_shoot",
    "source_probe_interpolate_contains_step_size",
    "source_probe_bvp_path_contains_solve",
    "source_probe_bvp_solve_contains_num_points",
    "lrw_geodesic_solver_endpoint_error_mean",
    "lrw_geodesic_solver_distance_error_mean",
    "lrw_geodesic_solver_geodesic_distance_error_mean",
    "lrw_geodesic_solver_energy_ratio_over_straight",
    "lrw_geodesic_solver_num_points_returned",
    "lrw_geodesic_solver_endpoint_error_mean",
    "lrw_geodesic_solver_distance_error_mean",
    "lrw_geodesic_solver_geodesic_distance_error_mean",
    "lrw_geodesic_solver_energy_ratio_over_straight",
    "lrw_geodesic_solver_num_points_returned",
    "lrw_slerp_endpoint_error_mean",
    "lrw_slerp_sphere_distance_error_mean",
    "lrw_slerp_sphere_radial_error_max",
    "lrw_slerp_vs_reference_relative_error_mean",
    "lrw_slerp_vs_reference_max_abs_error",
    "lrw_slerp_num_points_returned",
    "diagnostic_case_count",
    "diagnostic_call_failure_rate",
    "diagnostic_raw_endpoint_failure_rate",
    "diagnostic_raw_endpoint_pass_count",
    "diagnostic_clamped_endpoint_pass_count",
    "diagnostic_raw_endpoint_error_mean_min",
    "diagnostic_raw_endpoint_error_mean_max",
    "diagnostic_energy_improved_count",
    "diagnostic_energy_only_success_count",
    "diagnostic_overall_valid_count",
    "diagnostic_overall_valid_rate",
    "diagnostic_endpoint_tolerance_pass_rate",
    "diagnostic_boundary_condition_failure_count",
    "diagnostic_best_raw_endpoint_error",
    "diagnostic_best_raw_energy_ratio",
    "diagnostic_best_valid_score",
    "nan_rate",
    "inf_rate",
    "runtime_ms",
    "runtime_ms_mean",
    "runtime_ms_max",
    "failure_rate",
    "available",
    "public_symbol_count",
    "candidate_submodule_count",
    "api_class_count",
    "api_importable_class_count",
    "api_method_count",
    "api_signature_count",
    "sample_count",
    "regularization",
    "lrw_pullback_spd_violation_rate",
    "lrw_pullback_condition_number_mean",
    "lrw_pullback_condition_number_max",
    "lrw_vs_reference_same_shape",
    "lrw_vs_reference_mean_abs_error",
    "lrw_vs_reference_max_abs_error",
    "lrw_vs_reference_relative_frobenius_error_mean",
    "lrw_vs_reference_relative_frobenius_error_max",
]


CSV_COLUMNS = [
    "benchmark_id",
    "benchmark_layer",
    "manifold",
    "method",
    "status",
    "device",
    "dtype",
    "dimension",
    "batch_size",
    "num_points",
    "case_count",
    "failure_count",
    "failure_rate",
    "endpoint_error_mean",
    "distance_error_mean",
    "sphere_distance_error_mean",
    "energy_improvement",
    "length_improvement",
    "spd_violation_rate",
    "pullback_spd_violation_rate",
    "lrw_pullback_spd_violation_rate",
    "lrw_vs_reference_relative_frobenius_error_mean",
    "lrw_vs_reference_max_abs_error",
    "sample_count",
    "lrw_bvp_endpoint_error_mean",
    "lrw_bvp_pullback_energy_mean",
    "lrw_bvp_pullback_length_mean",
    "lrw_bvp_energy_ratio_over_straight",
    "lrw_bvp_num_points_returned",
    "lrw_euclidean_bvp_endpoint_error_mean",
    "lrw_euclidean_bvp_distance_error_mean",
    "lrw_euclidean_bvp_energy_ratio_over_straight",
    "lrw_euclidean_bvp_num_points_returned",
    "lrw_geodesic_solver_endpoint_error_mean",
    "lrw_geodesic_solver_distance_error_mean",
    "lrw_geodesic_solver_geodesic_distance_error_mean",
    "lrw_geodesic_solver_energy_ratio_over_straight",
    "lrw_geodesic_solver_num_points_returned",
    "lrw_slerp_endpoint_error_mean",
    "lrw_slerp_sphere_distance_error_mean",
    "lrw_slerp_sphere_radial_error_max",
    "lrw_slerp_vs_reference_relative_error_mean",
    "lrw_slerp_vs_reference_max_abs_error",
    "lrw_slerp_num_points_returned",
    "anatomy_call_count",
    "anatomy_successful_call_count",
    "anatomy_axis_candidate_count",
    "anatomy_axis_endpoint_valid_count",
    "anatomy_best_axis_endpoint_error",
    "anatomy_best_axis_distance_error",
    "anatomy_best_axis_energy_ratio",
    "anatomy_geodesic_distance_relative_error_mean",
    "scale_probe_case_count",
    "scale_probe_success_count",
    "scale_probe_failure_count",
    "scale_probe_best_endpoint_error",
    "scale_probe_best_movement_ratio",
    "scale_probe_distance_ratio_min",
    "scale_probe_distance_ratio_max",
    "scale_probe_movement_ratio_min",
    "scale_probe_movement_ratio_max",
    "scale_probe_distance_ratio_over_step_size_mean",
    "scale_probe_movement_ratio_over_step_size_mean",
    "diagnostic_case_count",
    "diagnostic_call_failure_rate",
    "diagnostic_raw_endpoint_failure_rate",
    "diagnostic_raw_endpoint_pass_count",
    "diagnostic_clamped_endpoint_pass_count",
    "diagnostic_raw_endpoint_error_mean_min",
    "diagnostic_raw_endpoint_error_mean_max",
    "diagnostic_energy_improved_count",
    "diagnostic_energy_only_success_count",
    "diagnostic_overall_valid_count",
    "diagnostic_overall_valid_rate",
    "diagnostic_endpoint_tolerance_pass_rate",
    "diagnostic_boundary_condition_failure_count",
    "diagnostic_best_raw_endpoint_error",
    "diagnostic_best_raw_energy_ratio",
    "diagnostic_best_valid_score",
    "nan_rate",
    "inf_rate",
    "runtime_ms",
    "runtime_ms_mean",
    "runtime_ms_max",
    "available",
    "module_name",
    "version",
    "api_class_count",
    "api_importable_class_count",
    "api_method_count",
    "api_signature_count",
    "sample_count",
    "regularization",
    "lrw_pullback_spd_violation_rate",
    "lrw_pullback_condition_number_mean",
    "lrw_pullback_condition_number_max",
    "lrw_vs_reference_same_shape",
    "lrw_vs_reference_mean_abs_error",
    "lrw_vs_reference_max_abs_error",
    "lrw_vs_reference_relative_frobenius_error_mean",
    "lrw_vs_reference_relative_frobenius_error_max",
]


def load_result_files(raw_dir: str | Path) -> list[dict[str, Any]]:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw result directory does not exist: {raw_dir}")

    results: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "benchmark_id" in data:
            data["_source_file"] = str(path)
            results.append(data)
    return results


def _metrics_blob(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    summary = result.get("summary")
    if isinstance(summary, dict):
        return summary
    return {}


def _get_value(result: dict[str, Any], key: str) -> Any:
    if key in result:
        return result[key]
    metrics = _metrics_blob(result)
    return metrics.get(key, "")


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for column in CSV_COLUMNS:
        row[column] = _get_value(result, column)

    # Stress benchmark stores scaling axes as lists at top-level.
    if row["dimension"] == "" and isinstance(result.get("dimensions"), list):
        row["dimension"] = ",".join(str(x) for x in result["dimensions"])
    if row["batch_size"] == "" and isinstance(result.get("batch_sizes"), list):
        row["batch_size"] = ",".join(str(x) for x in result["batch_sizes"])
    if row["num_points"] == "" and isinstance(result.get("num_points"), list):
        row["num_points"] = ",".join(str(x) for x in result["num_points"])

    # Some benchmark files use ambient_dimension or latent_dimension.
    if row["dimension"] == "":
        row["dimension"] = result.get("ambient_dimension", result.get("latent_dimension", ""))

    return row


def write_summary_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def _fmt(value: Any) -> str:
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


def _status_counts(results: list[dict[str, Any]]) -> tuple[int, int, int]:
    passed = sum(1 for r in results if r.get("status") == "success")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    failed = len(results) - passed - skipped
    return passed, skipped, failed


def make_markdown_report(results: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    passed, skipped, failed = _status_counts(results)

    lines: list[str] = []
    lines.append("# Latent Riemannian Benchmark Summary")
    lines.append("")
    lines.append(f"Generated at: `{generated_at}`")
    lines.append("")
    lines.append("## Overall Status")
    lines.append("")
    lines.append(f"- Total benchmark files: **{len(results)}**")
    lines.append(f"- Passed: **{passed}**")
    lines.append(f"- Skipped: **{skipped}**")
    lines.append(f"- Failed: **{failed}**")
    lines.append("")

    lines.append("## Benchmark Table")
    lines.append("")
    lines.append("| Benchmark | Layer | Manifold | Method | Status | Key metrics |")
    lines.append("|---|---|---|---|---|---|")
    for result in results:
        metrics = _metrics_blob(result)
        key_parts = []
        for key in IMPORTANT_METRIC_KEYS:
            if key in metrics:
                key_parts.append(f"`{key}={_fmt(metrics[key])}`")
        key_text = "<br>".join(key_parts[:6]) if key_parts else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(result.get("benchmark_id", "")),
                    _fmt(result.get("benchmark_layer", "")),
                    _fmt(result.get("manifold", result.get("target", ""))),
                    _fmt(result.get("method", result.get("target", ""))),
                    _fmt(result.get("status", "")),
                    key_text,
                ]
            )
            + " |"
        )
    lines.append("")

    skipped_results = [r for r in results if r.get("status") == "skipped"]
    if skipped_results:
        lines.append("## Skipped")
        lines.append("")
        for result in skipped_results:
            skip = result.get("skip", {})
            lines.append(f"- **{result.get('benchmark_id')}**: {skip.get('reason', 'skipped')}")
        lines.append("")

    failures = [r for r in results if r.get("failure", {}).get("has_failure") or r.get("status") == "failure"]
    lines.append("## Failures")
    lines.append("")
    if not failures:
        lines.append("No failures detected.")
    else:
        for result in failures:
            failure = result.get("failure", {})
            lines.append(
                f"- **{result.get('benchmark_id')}**: `{failure.get('failure_type')}` — {failure.get('message')}"
            )
    lines.append("")


    api_results = [r for r in results if r.get("method") == "lrw_api_introspection"]
    if api_results:
        lines.append("## LRW API Introspection")
        lines.append("")
        for result in api_results:
            api = result.get("api", {})
            lines.append(f"### {result.get('benchmark_id')}")
            lines.append("")
            lines.append(f"- Importable classes: `{_fmt(api.get('importable_class_count', ''))}` / `{_fmt(api.get('class_count', ''))}`")
            lines.append(f"- Public methods found: `{_fmt(api.get('method_count', ''))}`")
            lines.append(f"- Signatures found: `{_fmt(api.get('signature_count', ''))}`")
            lines.append("")
            lines.append("| Class | Init signature | Methods |")
            lines.append("|---|---|---|")
            for cls in api.get("classes", []):
                if not cls.get("importable"):
                    lines.append(f"| `{cls.get('class_path')}` | import failed | `{cls.get('error', '')}` |")
                    continue
                methods = []
                for method in cls.get("methods", []):
                    sig = method.get("signature") or ""
                    methods.append(f"`{method.get('name')}{sig}`")
                lines.append(
                    f"| `{cls.get('class_path')}` | `{cls.get('init_signature')}` | "
                    + "<br>".join(methods)
                    + " |"
                )
            lines.append("")

    lrw_pullback_results = [r for r in results if r.get("method") == "lrw_pullback_metric_tensor"]
    if lrw_pullback_results:
        lines.append("## LRW PullbackMetric Benchmark")
        lines.append("")
        for result in lrw_pullback_results:
            metrics = _metrics_blob(result)
            lines.append(f"### {result.get('benchmark_id')}")
            lines.append("")
            lines.append(f"- Status: `{_fmt(result.get('status', ''))}`")
            lines.append(f"- Sample count: `{_fmt(result.get('sample_count', metrics.get('sample_count', '')))}`")
            lines.append(f"- LRW SPD violation rate: `{_fmt(metrics.get('lrw_pullback_spd_violation_rate', ''))}`")
            lines.append(f"- LRW condition number mean: `{_fmt(metrics.get('lrw_pullback_condition_number_mean', ''))}`")
            lines.append(f"- Reference same shape: `{_fmt(metrics.get('lrw_vs_reference_same_shape', ''))}`")
            lines.append(f"- Reference relative Frobenius error mean: `{_fmt(metrics.get('lrw_vs_reference_relative_frobenius_error_mean', ''))}`")
            lines.append(f"- Runtime ms: `{_fmt(metrics.get('runtime_ms', ''))}`")
            if result.get("status") == "skipped":
                skip = result.get("skip", {})
                lines.append(f"- Skip reason: {skip.get('reason', 'skipped')}")
            lines.append("")


    lrw_bvp_results = [r for r in results if r.get("method") == "lrw_bvp_geodesic_path"]
    if lrw_bvp_results:
        lines.append("## LRW BVPSolver Benchmark")
        lines.append("")
        for result in lrw_bvp_results:
            metrics = _metrics_blob(result)
            lines.append(f"### {result.get('benchmark_id')}")
            lines.append("")
            lines.append(f"- Status: `{_fmt(result.get('status', ''))}`")
            lines.append(f"- Returned points: `{_fmt(metrics.get('lrw_bvp_num_points_returned', ''))}`")
            lines.append(f"- Endpoint error mean: `{_fmt(metrics.get('lrw_bvp_endpoint_error_mean', ''))}`")
            lines.append(f"- Pullback energy mean: `{_fmt(metrics.get('lrw_bvp_pullback_energy_mean', ''))}`")
            lines.append(f"- Pullback length mean: `{_fmt(metrics.get('lrw_bvp_pullback_length_mean', ''))}`")
            lines.append(f"- Energy ratio over straight: `{_fmt(metrics.get('lrw_bvp_energy_ratio_over_straight', ''))}`")
            lines.append(f"- Runtime ms: `{_fmt(metrics.get('runtime_ms', ''))}`")
            if result.get("status") == "skipped":
                skip = result.get("skip", {})
                lines.append(f"- Skip reason: {skip.get('reason', 'skipped')}")
            lines.append("")

    lrw_euclidean_bvp_results = [r for r in results if r.get("method") == "lrw_euclidean_bvp_geodesic_path"]
    if lrw_euclidean_bvp_results:
        lines.append("## LRW Euclidean BVPSolver Sanity Check")
        lines.append("")
        for result in lrw_euclidean_bvp_results:
            metrics = _metrics_blob(result)
            lines.append(f"### {result.get('benchmark_id')}")
            lines.append("")
            lines.append(f"- Status: `{_fmt(result.get('status', ''))}`")
            lines.append(f"- Returned points: `{_fmt(metrics.get('lrw_euclidean_bvp_num_points_returned', ''))}`")
            lines.append(f"- Endpoint error mean: `{_fmt(metrics.get('lrw_euclidean_bvp_endpoint_error_mean', ''))}`")
            lines.append(f"- Distance error mean: `{_fmt(metrics.get('lrw_euclidean_bvp_distance_error_mean', ''))}`")
            lines.append(f"- Energy ratio over straight: `{_fmt(metrics.get('lrw_euclidean_bvp_energy_ratio_over_straight', ''))}`")
            lines.append(f"- Runtime ms: `{_fmt(metrics.get('runtime_ms', ''))}`")
            if result.get("status") == "skipped":
                skip = result.get("skip", {})
                lines.append(f"- Skip reason: {skip.get('reason', 'skipped')}")
            lines.append("")

    lrw_geodesic_solver_results = [r for r in results if r.get("method") == "lrw_geodesic_solver_euclidean"]
    if lrw_geodesic_solver_results:
        lines.append("## LRW GeodesicSolver Euclidean Sanity Check")
        lines.append("")
        for result in lrw_geodesic_solver_results:
            metrics = _metrics_blob(result)
            lines.append(f"### {result.get('benchmark_id')}")
            lines.append("")
            lines.append(f"- Status: `{_fmt(result.get('status', ''))}`")
            lines.append(f"- Returned points: `{_fmt(metrics.get('lrw_geodesic_solver_num_points_returned', ''))}`")
            lines.append(f"- Endpoint error mean: `{_fmt(metrics.get('lrw_geodesic_solver_endpoint_error_mean', ''))}`")
            lines.append(f"- Path distance error mean: `{_fmt(metrics.get('lrw_geodesic_solver_distance_error_mean', ''))}`")
            lines.append(f"- geodesic_distance error mean: `{_fmt(metrics.get('lrw_geodesic_solver_geodesic_distance_error_mean', ''))}`")
            lines.append(f"- Energy ratio over straight: `{_fmt(metrics.get('lrw_geodesic_solver_energy_ratio_over_straight', ''))}`")
            lines.append(f"- Runtime ms: `{_fmt(metrics.get('runtime_ms', ''))}`")
            if result.get("status") == "skipped":
                skip = result.get("skip", {})
                lines.append(f"- Skip reason: {skip.get('reason', 'skipped')}")
            lines.append("")

    lrw_slerp_results = [r for r in results if r.get("method") == "lrw_slerp_path"]
    if lrw_slerp_results:
        lines.append("## LRW SLERP Sanity Check")
        lines.append("")
        for result in lrw_slerp_results:
            metrics = _metrics_blob(result)
            lines.append(f"### {result.get('benchmark_id')}")
            lines.append("")
            lines.append(f"- Status: `{_fmt(result.get('status', ''))}`")
            lines.append(f"- Function path: `{_fmt((result.get('lrw') or {}).get('function_path', ''))}`")
            lines.append(f"- Returned points: `{_fmt(metrics.get('lrw_slerp_num_points_returned', ''))}`")
            lines.append(f"- Endpoint error mean: `{_fmt(metrics.get('lrw_slerp_endpoint_error_mean', ''))}`")
            lines.append(f"- Sphere distance error mean: `{_fmt(metrics.get('lrw_slerp_sphere_distance_error_mean', ''))}`")
            lines.append(f"- Radial error max: `{_fmt(metrics.get('lrw_slerp_sphere_radial_error_max', ''))}`")
            lines.append(f"- Reference relative error mean: `{_fmt(metrics.get('lrw_slerp_vs_reference_relative_error_mean', ''))}`")
            lines.append(f"- Runtime ms: `{_fmt(metrics.get('runtime_ms', ''))}`")
            if result.get("status") == "skipped":
                skip = result.get("skip", {})
                lines.append(f"- Skip reason: {skip.get('reason', 'skipped')}")
            lines.append("")

    lrw_anatomy_results = [r for r in results if r.get("method") == "lrw_geodesic_output_anatomy"]
    if lrw_anatomy_results:
        lines.append("## LRW Geodesic Solver Output Anatomy")
        lines.append("")
        for result in lrw_anatomy_results:
            metrics = _metrics_blob(result)
            summary = result.get("summary", {})
            lines.append(f"### {result.get('benchmark_id')}")
            lines.append("")
            lines.append(f"- Status: `{_fmt(result.get('status', ''))}`")
            lines.append(f"- Call count: `{_fmt(summary.get('anatomy_call_count', metrics.get('anatomy_call_count', '')))}`")
            lines.append(f"- Successful calls: `{_fmt(summary.get('anatomy_successful_call_count', metrics.get('anatomy_successful_call_count', '')))}`")
            lines.append(f"- Axis candidate count: `{_fmt(summary.get('anatomy_axis_candidate_count', metrics.get('anatomy_axis_candidate_count', '')))}`")
            lines.append(f"- Axis endpoint-valid count: `{_fmt(summary.get('anatomy_axis_endpoint_valid_count', metrics.get('anatomy_axis_endpoint_valid_count', '')))}`")
            lines.append(f"- Best axis endpoint error: `{_fmt(summary.get('anatomy_best_axis_endpoint_error', metrics.get('anatomy_best_axis_endpoint_error', '')))}`")
            lines.append(f"- Best axis: `{_fmt(summary.get('anatomy_best_axis_call', ''))}` / `{_fmt(summary.get('anatomy_best_axis_name', ''))}`")
            lines.append(f"- Best axis distance error: `{_fmt(summary.get('anatomy_best_axis_distance_error', metrics.get('anatomy_best_axis_distance_error', '')))}`")
            lines.append(f"- Best axis energy ratio: `{_fmt(summary.get('anatomy_best_axis_energy_ratio', metrics.get('anatomy_best_axis_energy_ratio', '')))}`")
            lines.append(f"- geodesic_distance relative error mean: `{_fmt(summary.get('anatomy_geodesic_distance_relative_error_mean', metrics.get('anatomy_geodesic_distance_relative_error_mean', '')))}`")
            lines.append(f"- Verdict: `{_fmt(summary.get('anatomy_verdict', ''))}`")
            lines.append(f"- Runtime ms: `{_fmt(metrics.get('runtime_ms', ''))}`")
            calls = result.get("calls", [])
            if calls:
                lines.append("")
                lines.append("| Call | Status | Raw shape | Best candidate | Best endpoint | Best distance error | Energy ratio |")
                lines.append("|---|---|---|---|---:|---:|---:|")
                for call in calls:
                    candidates = [c for c in call.get('axis_candidates', []) if c.get('valid_shape')]
                    best = min(candidates, key=lambda c: float(c.get('endpoint_error_mean', 1e99))) if candidates else {}
                    lines.append(
                        f"| `{call.get('call_name')}` | `{call.get('status')}` | `{_fmt(call.get('raw_shape', ''))}` | "
                        f"`{_fmt(best.get('candidate_name', ''))}` | `{_fmt(best.get('endpoint_error_mean', ''))}` | "
                        f"`{_fmt(best.get('distance_error_mean', ''))}` | `{_fmt(best.get('energy_ratio_over_straight', ''))}` |"
                    )
            if result.get("status") == "skipped":
                skip = result.get("skip", {})
                lines.append(f"- Skip reason: {skip.get('reason', 'skipped')}")
            lines.append("")


    lrw_scale_probe_results = [r for r in results if r.get("method") == "lrw_solver_scale_probe"]
    if lrw_scale_probe_results:
        lines.append("## LRW Solver Step-Scale Probe")
        lines.append("")
        for result in lrw_scale_probe_results:
            metrics = _metrics_blob(result)
            summary = result.get("summary", {})
            lines.append(f"### {result.get('benchmark_id')}")
            lines.append("")
            lines.append(f"- Status: `{_fmt(result.get('status', ''))}`")
            lines.append(f"- Case count: `{_fmt(summary.get('scale_probe_case_count', metrics.get('scale_probe_case_count', '')))}`")
            lines.append(f"- Success count: `{_fmt(summary.get('scale_probe_success_count', metrics.get('scale_probe_success_count', '')))}`")
            lines.append(f"- Failure count: `{_fmt(summary.get('scale_probe_failure_count', metrics.get('scale_probe_failure_count', '')))}`")
            lines.append(f"- Best endpoint error: `{_fmt(summary.get('scale_probe_best_endpoint_error', metrics.get('scale_probe_best_endpoint_error', '')))}`")
            lines.append(f"- Best movement ratio: `{_fmt(summary.get('scale_probe_best_movement_ratio', metrics.get('scale_probe_best_movement_ratio', '')))}`")
            lines.append(f"- Distance ratio range: `{_fmt(summary.get('scale_probe_distance_ratio_min', metrics.get('scale_probe_distance_ratio_min', '')))}` → `{_fmt(summary.get('scale_probe_distance_ratio_max', metrics.get('scale_probe_distance_ratio_max', '')))}`")
            lines.append(f"- Movement ratio range: `{_fmt(summary.get('scale_probe_movement_ratio_min', metrics.get('scale_probe_movement_ratio_min', '')))}` → `{_fmt(summary.get('scale_probe_movement_ratio_max', metrics.get('scale_probe_movement_ratio_max', '')))}`")
            lines.append(f"- Mean distance_ratio / step_size: `{_fmt(summary.get('scale_probe_distance_ratio_over_step_size_mean', metrics.get('scale_probe_distance_ratio_over_step_size_mean', '')))}`")
            lines.append(f"- Mean movement_ratio / step_size: `{_fmt(summary.get('scale_probe_movement_ratio_over_step_size_mean', metrics.get('scale_probe_movement_ratio_over_step_size_mean', '')))}`")
            lines.append(f"- Verdict: `{_fmt(summary.get('scale_probe_verdict', ''))}`")
            lines.append(f"- Runtime ms: `{_fmt(metrics.get('runtime_ms', ''))}`")
            cases = result.get("cases", [])
            if cases:
                lines.append("")
                lines.append("| Case | n_steps | step_size | Status | Distance ratio | Movement ratio | Endpoint error | Ratio/step |")
                lines.append("|---|---:|---:|---|---:|---:|---:|---:|")
                for case in cases[:20]:
                    lines.append(
                        f"| `{case.get('case_id')}` | `{_fmt(case.get('n_steps', ''))}` | `{_fmt(case.get('step_size', ''))}` | "
                        f"`{case.get('status')}` | `{_fmt(case.get('distance_ratio', ''))}` | `{_fmt(case.get('movement_ratio', ''))}` | "
                        f"`{_fmt(case.get('endpoint_error_mean', ''))}` | `{_fmt(case.get('distance_ratio_over_step_size', ''))}` |"
                    )
            if result.get("status") == "skipped":
                skip = result.get("skip", {})
                lines.append(f"- Skip reason: {skip.get('reason', 'skipped')}")
            lines.append("")


    lrw_source_probe_results = [r for r in results if r.get("method") == "lrw_source_probe"]
    if lrw_source_probe_results:
        lines.append("## LRW Source / Implementation Fingerprint Probe")
        lines.append("")
        for result in lrw_source_probe_results:
            metrics = _metrics_blob(result)
            summary = result.get("summary", {})
            lines.append(f"### {result.get('benchmark_id')}")
            lines.append("")
            lines.append(f"- Status: `{_fmt(result.get('status', ''))}`")
            lines.append(f"- Target count: `{_fmt(summary.get('source_probe_target_count', metrics.get('source_probe_target_count', '')))}`")
            lines.append(f"- Success count: `{_fmt(summary.get('source_probe_success_count', metrics.get('source_probe_success_count', '')))}`")
            lines.append(f"- Failure count: `{_fmt(summary.get('source_probe_failure_count', metrics.get('source_probe_failure_count', '')))}`")
            lines.append(f"- step_size references: `{_fmt(summary.get('source_probe_step_size_reference_count', metrics.get('source_probe_step_size_reference_count', '')))}`")
            lines.append(f"- shoot references: `{_fmt(summary.get('source_probe_shoot_reference_count', metrics.get('source_probe_shoot_reference_count', '')))}`")
            lines.append(f"- num_points references: `{_fmt(summary.get('source_probe_num_points_reference_count', metrics.get('source_probe_num_points_reference_count', '')))}`")
            lines.append(f"- geodesic_distance contains step_size: `{_fmt(summary.get('source_probe_geodesic_distance_contains_step_size', metrics.get('source_probe_geodesic_distance_contains_step_size', '')))}`")
            lines.append(f"- interpolate contains shoot: `{_fmt(summary.get('source_probe_interpolate_contains_shoot', metrics.get('source_probe_interpolate_contains_shoot', '')))}`")
            lines.append(f"- interpolate contains step_size: `{_fmt(summary.get('source_probe_interpolate_contains_step_size', metrics.get('source_probe_interpolate_contains_step_size', '')))}`")
            lines.append(f"- BVPSolver.geodesic_path contains solve: `{_fmt(summary.get('source_probe_bvp_path_contains_solve', metrics.get('source_probe_bvp_path_contains_solve', '')))}`")
            lines.append(f"- BVPSolver.solve contains num_points: `{_fmt(summary.get('source_probe_bvp_solve_contains_num_points', metrics.get('source_probe_bvp_solve_contains_num_points', '')))}`")
            lines.append(f"- Verdict: `{_fmt(summary.get('source_probe_verdict', ''))}`")
            lines.append(f"- Runtime ms: `{_fmt(metrics.get('runtime_ms', ''))}`")
            entries = result.get("source_entries", [])
            if entries:
                lines.append("")
                lines.append("| Target | Status | File | Line | Lines | SHA16 | Key flags |")
                lines.append("|---|---|---|---:|---:|---|---|")
                for entry in entries:
                    flags = []
                    for key, label in [
                        ("contains_step_size", "step_size"),
                        ("contains_n_steps", "n_steps"),
                        ("contains_shoot", "shoot"),
                        ("contains_solve", "solve"),
                        ("contains_num_points", "num_points"),
                        ("contains_metric_tensor", "metric_tensor"),
                    ]:
                        if entry.get(key):
                            flags.append(label)
                    lines.append(
                        f"| `{entry.get('target_id')}` | `{entry.get('status')}` | `{_fmt(entry.get('source_file', ''))}` | "
                        f"`{_fmt(entry.get('source_start_line', ''))}` | `{_fmt(entry.get('source_line_count', ''))}` | "
                        f"`{_fmt(entry.get('source_sha256_16', ''))}` | `{', '.join(flags)}` |"
                    )
            if result.get("status") == "skipped":
                skip = result.get("skip", {})
                lines.append(f"- Skip reason: {skip.get('reason', 'skipped')}")
            lines.append("")

    lrw_bvp_diagnostic_results = [r for r in results if r.get("method") == "lrw_bvp_diagnostic"]
    if lrw_bvp_diagnostic_results:
        lines.append("## LRW BVP Diagnostic")
        lines.append("")
        for result in lrw_bvp_diagnostic_results:
            metrics = _metrics_blob(result)
            summary = result.get("summary", {})
            lines.append(f"### {result.get('benchmark_id')}")
            lines.append("")
            lines.append(f"- Status: `{_fmt(result.get('status', ''))}`")
            lines.append(f"- Case count: `{_fmt(summary.get('case_count', metrics.get('diagnostic_case_count', '')))}`")
            lines.append(f"- Call failure rate: `{_fmt(summary.get('call_failure_rate', metrics.get('diagnostic_call_failure_rate', '')))}`")
            lines.append(f"- Raw endpoint failure rate: `{_fmt(summary.get('raw_endpoint_failure_rate', metrics.get('diagnostic_raw_endpoint_failure_rate', '')))}`")
            lines.append(f"- Raw endpoint pass count: `{_fmt(summary.get('raw_endpoint_pass_count', metrics.get('diagnostic_raw_endpoint_pass_count', '')))}`")
            lines.append(f"- Clamped endpoint pass count: `{_fmt(summary.get('clamped_endpoint_pass_count', metrics.get('diagnostic_clamped_endpoint_pass_count', '')))}`")
            lines.append(f"- Raw endpoint error range: `{_fmt(summary.get('raw_endpoint_error_mean_min', ''))}` → `{_fmt(summary.get('raw_endpoint_error_mean_max', ''))}`")
            lines.append(f"- Energy-improved cases: `{_fmt(summary.get('energy_improved_count', metrics.get('diagnostic_energy_improved_count', '')))}`")
            lines.append(f"- Energy-only successes: `{_fmt(summary.get('energy_only_success_count', metrics.get('diagnostic_energy_only_success_count', '')))}`")
            lines.append(f"- Overall valid cases: `{_fmt(summary.get('overall_valid_count', metrics.get('diagnostic_overall_valid_count', '')))}`")
            lines.append(f"- Overall valid rate: `{_fmt(summary.get('overall_valid_rate', metrics.get('diagnostic_overall_valid_rate', '')))}`")
            best_endpoint = summary.get('best_by_endpoint') or {}
            best_energy = summary.get('best_by_energy') or {}
            best_valid = summary.get('best_by_valid_score') or {}
            lines.append(f"- Best endpoint case: `{_fmt(best_endpoint.get('case_id', ''))}` endpoint=`{_fmt(best_endpoint.get('raw_endpoint_error_mean', ''))}` energy_ratio=`{_fmt(best_endpoint.get('raw_energy_ratio_over_straight', ''))}`")
            lines.append(f"- Best energy case: `{_fmt(best_energy.get('case_id', ''))}` endpoint=`{_fmt(best_energy.get('raw_endpoint_error_mean', ''))}` energy_ratio=`{_fmt(best_energy.get('raw_energy_ratio_over_straight', ''))}`")
            lines.append(f"- Best validity-score case: `{_fmt(best_valid.get('case_id', ''))}` score=`{_fmt(best_valid.get('valid_geodesic_score', ''))}` overall_valid=`{_fmt(best_valid.get('overall_valid', ''))}`")
            lines.append(f"- Runtime ms: `{_fmt(metrics.get('runtime_ms', ''))}`")
            cases = result.get("cases", [])[:8]
            if cases:
                lines.append("")
                lines.append("| Case | Method | Params | Status | Raw endpoint | Energy ratio | Overall valid | Valid score |")
                lines.append("|---|---|---|---|---:|---:|---:|---:|")
                for case in cases:
                    compact = case.get("compact", {})
                    params = case.get("params", {})
                    params_str = f"n_steps={params.get('solver_n_steps')}, lr={params.get('solver_lr')}, max_iter={params.get('solver_max_iter')}"
                    lines.append(
                        f"| `{case.get('case_id')}` | `{case.get('method_name')}` | `{params_str}` | `{case.get('status')}` | "
                        f"`{_fmt(compact.get('raw_endpoint_error_mean', ''))}` | `{_fmt(compact.get('raw_energy_ratio_over_straight', ''))}` | "
                        f"`{_fmt(compact.get('overall_valid', ''))}` | `{_fmt(compact.get('valid_geodesic_score', ''))}` |"
                    )
            lines.append("")

    stress = [r for r in results if r.get("benchmark_layer") == "stress"]
    if stress:
        lines.append("## Stress Summary")
        lines.append("")
        for result in stress:
            summary = result.get("summary", {})
            lines.append(f"### {result.get('benchmark_id')}")
            lines.append("")
            lines.append(f"- Case count: `{_fmt(summary.get('case_count', ''))}`")
            lines.append(f"- Failure rate: `{_fmt(summary.get('failure_rate', ''))}`")
            lines.append(f"- Runtime mean ms: `{_fmt(summary.get('runtime_ms_mean', ''))}`")
            lines.append(f"- Runtime max ms: `{_fmt(summary.get('runtime_ms_max', ''))}`")
            lines.append(f"- Max distance error: `{_fmt(summary.get('distance_error_mean_max', ''))}`")
            lines.append("")

    lines.append("## CSV Columns")
    lines.append("")
    lines.append("A machine-readable CSV version is written next to this Markdown report.")
    lines.append("")
    lines.append("```txt")
    lines.append(", ".join(CSV_COLUMNS))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write_markdown_report(markdown: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")


def generate_report(raw_dir: str | Path, report_dir: str | Path) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    report_dir = Path(report_dir)
    results = load_result_files(raw_dir)
    rows = [flatten_result(result) for result in results]

    csv_path = report_dir / "summary.csv"
    md_path = report_dir / "summary.md"
    write_summary_csv(rows, csv_path)
    markdown = make_markdown_report(results, rows)
    write_markdown_report(markdown, md_path)

    passed, skipped, failed = _status_counts(results)
    return {
        "status": "success" if failed == 0 else "failure",
        "raw_dir": str(raw_dir),
        "report_dir": str(report_dir),
        "markdown_path": str(md_path),
        "csv_path": str(csv_path),
        "benchmark_file_count": len(results),
        "passed_count": passed,
        "skipped_count": skipped,
        "failed_count": failed,
    }
