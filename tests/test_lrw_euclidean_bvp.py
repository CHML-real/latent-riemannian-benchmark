from __future__ import annotations

import sys
import types

import torch

from lrbench.methods.lrw_euclidean_bvp import build_lrw_euclidean_metric, lrw_euclidean_bvp_path_metrics


def test_lrw_euclidean_metric_tensor_is_identity():
    class BaseMetric:
        pass

    metric = build_lrw_euclidean_metric(BaseMetric)
    z = torch.randn(5, 3)
    g = metric.metric_tensor(z)
    assert g.shape == (5, 3, 3)
    expected = torch.eye(3).expand(5, 3, 3)
    assert torch.allclose(g, expected)
    assert torch.allclose(metric.geodesic_acceleration(z, z), torch.zeros_like(z))


def test_lrw_euclidean_bvp_path_metrics_for_line_success():
    z0 = torch.tensor([[-1.0, 0.0]])
    z1 = torch.tensor([[1.0, 0.0]])
    t = torch.linspace(0, 1, 5).view(5, 1, 1)
    path = (1 - t) * z0 + t * z1
    metrics = lrw_euclidean_bvp_path_metrics(path, z0, z1)
    assert metrics["lrw_euclidean_bvp_endpoint_error_mean"] == 0.0
    assert metrics["lrw_euclidean_bvp_distance_error_mean"] < 1e-6
    assert metrics["lrw_euclidean_bvp_energy_ratio_over_straight"] == 1.0


def test_lrw_euclidean_bvp_runner_skips_when_lrw_missing(tmp_path, monkeypatch):
    from lrbench.runners.run_lrw_euclidean_bvp import run

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
                "benchmark_id: lrw_euclidean_bvp_test_missing",
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


def test_lrw_euclidean_bvp_runner_with_fake_lrw_success(tmp_path, monkeypatch):
    from lrbench.runners.run_lrw_euclidean_bvp import run

    lrw_module = types.ModuleType("lrw")
    geodesic_module = types.ModuleType("lrw.geodesic")
    bvp_module = types.ModuleType("lrw.geodesic.bvp")
    metric_module = types.ModuleType("lrw.metric")
    base_module = types.ModuleType("lrw.metric.base")

    class FakeBaseMetric:
        pass

    class FakeBVPSolver:
        def __init__(self, metric, n_steps=20, step_size=0.05, lr=0.1, max_iter=50, tol=0.001):
            self.metric = metric
            self.n_steps = n_steps

        def geodesic_path(self, z0, z1, n_points=10):
            t = torch.linspace(0, 1, n_points, device=z0.device, dtype=z0.dtype).view(n_points, 1, 1)
            path = (1 - t) * z0 + t * z1
            return path, {"converged": True, "iterations": 1}

    bvp_module.BVPSolver = FakeBVPSolver
    base_module.RiemannianMetric = FakeBaseMetric
    monkeypatch.setitem(sys.modules, "lrw", lrw_module)
    monkeypatch.setitem(sys.modules, "lrw.geodesic", geodesic_module)
    monkeypatch.setitem(sys.modules, "lrw.geodesic.bvp", bvp_module)
    monkeypatch.setitem(sys.modules, "lrw.metric", metric_module)
    monkeypatch.setitem(sys.modules, "lrw.metric.base", base_module)

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "\n".join(
            [
                "benchmark_id: lrw_euclidean_bvp_test_fake",
                "device: cpu",
                "dtype: float32",
                "dimension: 2",
                "batch_size: 1",
                "n_points: 6",
                "solver_max_iter: 2",
                "endpoint_tolerance: 0.0001",
                "distance_tolerance: 0.0001",
                "energy_ratio_tolerance: 1.0001",
                f"output_dir: {tmp_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    result = run(str(cfg))
    assert result["status"] == "success"
    assert result["metrics"]["available"] is True
    assert result["metrics"]["lrw_euclidean_bvp_endpoint_error_mean"] == 0.0
    assert result["metrics"]["lrw_euclidean_bvp_num_points_returned"] == 6
