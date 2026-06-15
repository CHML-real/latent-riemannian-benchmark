from __future__ import annotations

import sys
import types

import torch

from lrbench.manifolds.sphere import normalize_to_sphere
from lrbench.methods.lrw_slerp import lrw_slerp_metrics
from lrbench.methods.sphere_great_circle import sphere_great_circle


def test_lrw_slerp_metrics_for_reference_success():
    z0 = normalize_to_sphere(torch.tensor([[1.0, 0.0, 0.0]]))
    z1 = normalize_to_sphere(torch.tensor([[0.0, 1.0, 0.0]]))
    path = sphere_great_circle(z0, z1, num_points=8)
    metrics = lrw_slerp_metrics(path, z0, z1, reference_path=path)
    assert metrics["lrw_slerp_endpoint_error_mean"] < 1e-6
    assert metrics["lrw_slerp_sphere_distance_error_mean"] < 1e-5
    assert metrics["lrw_slerp_sphere_radial_error_max"] < 1e-6
    assert metrics["lrw_slerp_vs_reference_relative_error_mean"] == 0.0
    assert metrics["lrw_slerp_num_points_returned"] == 8


def test_lrw_slerp_runner_skips_when_lrw_missing(tmp_path, monkeypatch):
    from lrbench.runners.run_lrw_slerp import run

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("lrw"):
            raise ImportError("mock missing lrw")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "\n".join(
            [
                "benchmark_id: lrw_slerp_test_missing",
                "device: cpu",
                "dtype: float32",
                "dimension: 3",
                "batch_size: 1",
                "n_points: 5",
                f"output_dir: {tmp_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    result = run(str(cfg))
    assert result["status"] == "skipped"
    assert result["skip"]["is_skipped"] is True


def test_lrw_slerp_runner_with_fake_lrw_success(tmp_path, monkeypatch):
    from lrbench.runners.run_lrw_slerp import run

    lrw_module = types.ModuleType("lrw")
    geodesic_module = types.ModuleType("lrw.geodesic")

    def fake_slerp_path(z0, z1, n_points=10):
        return sphere_great_circle(z0, z1, num_points=n_points)

    geodesic_module.slerp_path = fake_slerp_path
    monkeypatch.setitem(sys.modules, "lrw", lrw_module)
    monkeypatch.setitem(sys.modules, "lrw.geodesic", geodesic_module)

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "\n".join(
            [
                "benchmark_id: lrw_slerp_test_fake",
                "device: cpu",
                "dtype: float32",
                "dimension: 3",
                "batch_size: 1",
                "n_points: 7",
                "endpoint_tolerance: 0.0001",
                "sphere_distance_tolerance: 0.0001",
                "radial_tolerance: 0.0001",
                "reference_error_tolerance: 0.0001",
                f"output_dir: {tmp_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    result = run(str(cfg))
    assert result["status"] == "success"
    assert result["metrics"]["available"] is True
    assert result["metrics"]["lrw_slerp_endpoint_error_mean"] < 1e-6
    assert result["metrics"]["lrw_slerp_vs_reference_relative_error_mean"] < 1e-6
    assert result["metrics"]["lrw_slerp_num_points_returned"] == 7


def test_report_includes_lrw_slerp_section(tmp_path):
    from lrbench.reports.summary import generate_report

    raw = tmp_path / "raw"
    rep = tmp_path / "reports"
    raw.mkdir()
    (raw / "lrw_slerp_001.json").write_text(
        """
        {
          "benchmark_id": "lrw_slerp_001",
          "benchmark_layer": "adapter",
          "manifold": "sphere",
          "method": "lrw_slerp_path",
          "status": "success",
          "lrw": {"function_path": "lrw.geodesic.slerp_path"},
          "metrics": {
            "lrw_slerp_endpoint_error_mean": 0.0,
            "lrw_slerp_sphere_distance_error_mean": 0.0,
            "lrw_slerp_sphere_radial_error_max": 0.0,
            "lrw_slerp_vs_reference_relative_error_mean": 0.0,
            "lrw_slerp_num_points_returned": 5,
            "runtime_ms": 1.0
          },
          "failure": {"has_failure": false, "failure_type": null, "message": null},
          "skip": {"is_skipped": false, "reason": null}
        }
        """,
        encoding="utf-8",
    )
    generate_report(raw, rep)
    md = (rep / "summary.md").read_text(encoding="utf-8")
    assert "LRW SLERP Sanity Check" in md
    assert "lrw_slerp_001" in md
