from __future__ import annotations

import sys
import types

import torch

from lrbench.methods.lrw_solver_scale_probe import make_scale_probe_cases, run_lrw_solver_scale_probe


def test_make_scale_probe_cases_grid_order():
    cases = make_scale_probe_cases([1, 2], [0.1, 1.0])
    assert [c.case_id for c in cases] == ["case_001", "case_002", "case_003", "case_004"]
    assert cases[0].n_steps == 1
    assert cases[0].step_size == 0.1
    assert cases[-1].n_steps == 2
    assert cases[-1].step_size == 1.0


def test_scale_probe_runner_skips_when_lrw_missing(tmp_path, monkeypatch):
    from lrbench.runners.run_lrw_solver_scale_probe import run

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
                "benchmark_id: lrw_solver_scale_probe_test_missing",
                "device: cpu",
                "dtype: float32",
                "dimension: 2",
                "batch_size: 1",
                "n_points: 5",
                "n_steps_values: [1]",
                "step_size_values: [0.01]",
                f"output_dir: {tmp_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    result = run(str(cfg))
    assert result["status"] == "skipped"
    assert result["metrics"]["scale_probe_case_count"] == 0


def test_scale_probe_with_fake_lrw_exposes_step_size_scaling(monkeypatch):
    lrw_module = types.ModuleType("lrw")
    geodesic_module = types.ModuleType("lrw.geodesic")
    solver_module = types.ModuleType("lrw.geodesic.solver")
    metric_module = types.ModuleType("lrw.metric")
    base_module = types.ModuleType("lrw.metric.base")

    class FakeBaseMetric:
        pass

    class FakeScaledGeodesicSolver:
        def __init__(self, metric, n_steps=100, step_size=0.01):
            self.metric = metric
            self.n_steps = n_steps
            self.step_size = step_size

        def interpolate(self, z0, z1, n_points=10):
            # Simulate local rollout: only moves step_size fraction toward z1.
            t = torch.linspace(0, self.step_size, n_points, device=z0.device, dtype=z0.dtype).view(n_points, 1, 1)
            return (1 - t) * z0 + t * z1

        def geodesic_distance(self, z0, z1):
            return torch.linalg.norm(z1 - z0, dim=-1) * self.step_size

    solver_module.GeodesicSolver = FakeScaledGeodesicSolver
    base_module.RiemannianMetric = FakeBaseMetric
    monkeypatch.setitem(sys.modules, "lrw", lrw_module)
    monkeypatch.setitem(sys.modules, "lrw.geodesic", geodesic_module)
    monkeypatch.setitem(sys.modules, "lrw.geodesic.solver", solver_module)
    monkeypatch.setitem(sys.modules, "lrw.metric", metric_module)
    monkeypatch.setitem(sys.modules, "lrw.metric.base", base_module)

    z0 = torch.tensor([[0.0, 0.0]])
    z1 = torch.tensor([[1.0, 0.0]])
    rows, info, summary = run_lrw_solver_scale_probe(z0, z1, n_points=5, n_steps_values=[1], step_size_values=[0.01, 1.0])
    assert info["available"] is True
    assert len(rows) == 2
    assert abs(rows[0]["distance_ratio"] - 0.01) < 1e-7
    assert abs(rows[1]["distance_ratio"] - 1.0) < 1e-7
    assert summary["scale_probe_success_count"] == 2
    assert summary["scale_probe_best_endpoint_error"] == 0.0
    assert "distance_scales_like_step_size" in summary["scale_probe_verdict"]
