from __future__ import annotations

from pathlib import Path

from lrbench.methods.lrw_introspection import inspect_class, introspect_lrw_api
from lrbench.reports.summary import generate_report
from lrbench.runners.run_lrw_introspection import run_lrw_introspection


class FakeMetric:
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def metric_tensor(self, z):
        return z

    def geodesic_acceleration(self, z, v):
        return z * 0


def test_inspect_class_finds_init_and_methods() -> None:
    info = inspect_class("tests.test_lrw_introspection.FakeMetric")
    assert info["importable"] is True
    assert info["class_name"] == "FakeMetric"
    assert "scale" in info["init_signature"]
    method_names = {m["name"] for m in info["methods"]}
    assert "metric_tensor" in method_names
    assert "geodesic_acceleration" in method_names


def test_introspection_reports_missing_class() -> None:
    result = introspect_lrw_api(["not_a_real_package.NotAClass"])
    assert result["all_importable"] is False
    assert result["importable_class_count"] == 0
    assert result["classes"][0]["importable"] is False


def test_lrw_introspection_runner_writes_json(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    out = tmp_path / "raw" / "lrw_api_introspection_001.json"
    cfg.write_text(
        "benchmark_id: lrw_api_introspection_001\n"
        "benchmark_layer: adapter\n"
        "target: lrw_api\n"
        "method: lrw_api_introspection\n"
        f"output_path: {out.as_posix()}\n"
        "class_paths:\n"
        "  - tests.test_lrw_introspection.FakeMetric\n",
        encoding="utf-8",
    )
    result = run_lrw_introspection(str(cfg))
    assert result["status"] == "success"
    assert out.exists()
    assert result["metrics"]["api_method_count"] >= 2


def test_report_includes_lrw_api_section(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    reports = tmp_path / "reports"
    raw.mkdir()
    (raw / "lrw_api_introspection_001.json").write_text(
        '''{
          "benchmark_id": "lrw_api_introspection_001",
          "benchmark_layer": "adapter",
          "target": "lrw_api",
          "method": "lrw_api_introspection",
          "status": "success",
          "metrics": {"api_class_count": 1, "api_importable_class_count": 1, "api_method_count": 2, "api_signature_count": 2, "runtime_ms": 1.0},
          "api": {"class_count": 1, "importable_class_count": 1, "method_count": 2, "signature_count": 2, "classes": [{"class_path": "x.Foo", "importable": true, "init_signature": "(self)", "methods": [{"name": "bar", "signature": "(self)", "kind": "function"}]}]},
          "failure": {"has_failure": false, "failure_type": null, "message": null}
        }''',
        encoding="utf-8",
    )
    generate_report(raw, reports)
    md = (reports / "summary.md").read_text(encoding="utf-8")
    assert "LRW API Introspection" in md
    assert "x.Foo" in md
