from __future__ import annotations

import json
from pathlib import Path

from lrbench.reports.lrw_evidence import build_evidence_rows, generate_lrw_evidence_report, make_evidence_markdown


def _result(benchmark_id: str, method: str, status: str = "success", metrics: dict | None = None, **extra):
    data = {
        "benchmark_id": benchmark_id,
        "benchmark_layer": "adapter",
        "method": method,
        "manifold": extra.pop("manifold", "test"),
        "status": status,
        "metrics": metrics or {},
        "failure": {"has_failure": status == "failure", "failure_type": "TEST", "message": "test" if status == "failure" else None},
    }
    data.update(extra)
    return data


def test_build_evidence_rows_contains_core_components() -> None:
    rows = build_evidence_rows([
        _result("lrw_probe_001", "lrw_package_probe", metrics={"available": True}),
        _result("lrw_pullback_metric_001", "lrw_pullback_metric_tensor", metrics={"lrw_vs_reference_relative_frobenius_error_mean": 0.0}),
        _result("lrw_slerp_001", "lrw_slerp_path", metrics={"lrw_slerp_endpoint_error_mean": 0.0}),
        _result("lrw_geodesic_solver_euclidean_001", "lrw_geodesic_solver_euclidean", status="failure", metrics={"lrw_geodesic_solver_endpoint_error_mean": 0.1}),
    ])
    components = {row.component for row in rows}
    assert "PullbackMetric.metric_tensor" in components
    assert "slerp_path" in components
    assert "GeodesicSolver.interpolate / geodesic_distance" in components


def test_make_evidence_markdown_has_failure_modes() -> None:
    rows = build_evidence_rows([])
    md = make_evidence_markdown([], rows)
    assert "LRW Adapter Evidence Report" in md
    assert "GEO_F01_ENDPOINT_MISS" in md
    assert "DIST_F01_STEP_SIZE_SCALED" in md
    assert "Evidence Matrix" in md


def test_generate_lrw_evidence_report_writes_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    reports = tmp_path / "reports"
    raw.mkdir()
    sample = _result("lrw_slerp_001", "lrw_slerp_path", metrics={"lrw_slerp_endpoint_error_mean": 0.0})
    (raw / "lrw_slerp_001.json").write_text(json.dumps(sample), encoding="utf-8")
    out = generate_lrw_evidence_report(raw, reports)
    assert out["status"] == "success"
    assert (reports / "lrw_evidence_report.md").exists()
    assert (reports / "lrw_evidence_matrix.csv").exists()
