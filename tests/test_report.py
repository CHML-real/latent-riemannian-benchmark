from __future__ import annotations

import json
from pathlib import Path

from lrbench.reports.summary import flatten_result, generate_report, load_result_files


def test_flatten_result_reads_metrics() -> None:
    result = {
        "benchmark_id": "x",
        "benchmark_layer": "analytic",
        "manifold": "euclidean",
        "method": "euclidean_line",
        "status": "success",
        "device": "cpu",
        "dtype": "float32",
        "dimension": 8,
        "batch_size": 16,
        "num_points": 32,
        "metrics": {
            "endpoint_error_mean": 0.0,
            "distance_error_mean": 1e-8,
            "runtime_ms": 0.5,
        },
    }
    row = flatten_result(result)
    assert row["benchmark_id"] == "x"
    assert row["endpoint_error_mean"] == 0.0
    assert row["distance_error_mean"] == 1e-8
    assert row["runtime_ms"] == 0.5


def test_generate_report_writes_markdown_and_csv(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    report_dir = tmp_path / "reports"
    raw_dir.mkdir()

    result = {
        "benchmark_id": "analytic_euclidean_001",
        "benchmark_layer": "analytic",
        "manifold": "euclidean",
        "method": "euclidean_line",
        "status": "success",
        "device": "cpu",
        "dtype": "float32",
        "dimension": 8,
        "batch_size": 128,
        "num_points": 32,
        "metrics": {
            "endpoint_error_mean": 0.0,
            "distance_error_mean": 4.2e-8,
            "nan_rate": 0.0,
            "inf_rate": 0.0,
            "runtime_ms": 0.7,
        },
        "failure": {"has_failure": False, "failure_type": None, "message": None},
    }
    (raw_dir / "analytic_euclidean_001.json").write_text(json.dumps(result), encoding="utf-8")

    summary = generate_report(raw_dir, report_dir)
    assert summary["status"] == "success"
    assert summary["benchmark_file_count"] == 1
    assert (report_dir / "summary.md").exists()
    assert (report_dir / "summary.csv").exists()
    assert "Latent Riemannian Benchmark Summary" in (report_dir / "summary.md").read_text(encoding="utf-8")
    assert "analytic_euclidean_001" in (report_dir / "summary.csv").read_text(encoding="utf-8")


def test_load_result_files_ignores_non_json(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "note.txt").write_text("ignore", encoding="utf-8")
    (raw_dir / "valid.json").write_text(json.dumps({"benchmark_id": "ok"}), encoding="utf-8")
    results = load_result_files(raw_dir)
    assert len(results) == 1
    assert results[0]["benchmark_id"] == "ok"
