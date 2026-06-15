from __future__ import annotations

import sys
import types

import torch

from lrbench.methods.lrw_geodesic_solver import lrw_geodesic_solver_euclidean_metrics


def test_lrw_geodesic_solver_metrics_for_line_success():
    z0 = torch.tensor([[-1.0, 0.0]])
    z1 = torch.tensor([[1.0, 0.0]])
    t = torch.linspace(0, 1, 5).view(5, 1, 1)
    path = (1 - t) * z0 + t * z1
    distance = torch.tensor([2.0])
    metrics = lrw_geodesic_solver_euclidean_metrics(path, distance, z0, z1)
    assert metrics["lrw_geodesic_solver_endpoint_error_mean"] == 0.0
    assert metrics["lrw_geodesic_solver_distance_error_mean"] < 1e-6
    assert metrics["lrw_geodesic_solver_geodesic_distance_error_mean"] < 1e-6
    assert metrics["lrw_geodesic_solver_energy_ratio_over_straight"] == 1.0


def test_lrw_geodesic_solver_runner_skips_when_lrw_missing(tmp_path, monkeypatch):
    from lrbench.runners.run_lrw_geodesic_solver import run

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
                "benchmark_id: lrw_geodesic_solver_test_missing",
                "device: cpu",
                "dtype: float32",
                "dimension: 2",
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


def test_lrw_geodesic_solver_runner_with_fake_lrw_success(tmp_path, monkeypatch):
    from lrbench.runners.run_lrw_geodesic_solver import run

    lrw_module = types.ModuleType("lrw")
    geodesic_module = types.ModuleType("lrw.geodesic")
    solver_module = types.ModuleType("lrw.geodesic.solver")
    metric_module = types.ModuleType("lrw.metric")
    base_module = types.ModuleType("lrw.metric.base")

    class FakeBaseMetric:
        pass

    class FakeGeodesicSolver:
        def __init__(self, metric, n_steps=100, step_size=0.01):
            self.metric = metric
            self.n_steps = n_steps
            self.step_size = step_size

        def interpolate(self, z0, z1, n_points=10):
            t = torch.linspace(0, 1, n_points, device=z0.device, dtype=z0.dtype).view(n_points, 1, 1)
            return (1 - t) * z0 + t * z1

        def geodesic_distance(self, z0, z1):
            return torch.linalg.norm(z1 - z0, dim=-1)

    solver_module.GeodesicSolver = FakeGeodesicSolver
    base_module.RiemannianMetric = FakeBaseMetric
    monkeypatch.setitem(sys.modules, "lrw", lrw_module)
    monkeypatch.setitem(sys.modules, "lrw.geodesic", geodesic_module)
    monkeypatch.setitem(sys.modules, "lrw.geodesic.solver", solver_module)
    monkeypatch.setitem(sys.modules, "lrw.metric", metric_module)
    monkeypatch.setitem(sys.modules, "lrw.metric.base", base_module)

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "\n".join(
            [
                "benchmark_id: lrw_geodesic_solver_test_fake",
                "device: cpu",
                "dtype: float32",
                "dimension: 2",
                "batch_size: 1",
                "n_points: 6",
                "endpoint_tolerance: 0.0001",
                "distance_tolerance: 0.0001",
                "geodesic_distance_tolerance: 0.0001",
                "energy_ratio_tolerance: 1.0001",
                f"output_dir: {tmp_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    result = run(str(cfg))
    assert result["status"] == "success"
    assert result["metrics"]["available"] is True
    assert result["metrics"]["lrw_geodesic_solver_endpoint_error_mean"] == 0.0
    assert result["metrics"]["lrw_geodesic_solver_geodesic_distance_error_mean"] < 1e-6
    assert result["metrics"]["lrw_geodesic_solver_num_points_returned"] == 6
