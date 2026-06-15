from __future__ import annotations

import sys
import types

import torch

from lrbench.methods.lrw_bvp_solver import normalize_lrw_path


def test_normalize_lrw_path_accepts_single_path():
    z0 = torch.tensor([[-1.0, 0.0]])
    z1 = torch.tensor([[1.0, 0.0]])
    raw = torch.stack([z0[0], z1[0]], dim=0)
    path = normalize_lrw_path(raw, z0, z1, requested_n_points=2)
    assert path.shape == (2, 1, 2)
    assert torch.allclose(path[0], z0)
    assert torch.allclose(path[-1], z1)


def test_normalize_lrw_path_accepts_batch_first_path():
    z0 = torch.tensor([[-1.0, 0.0], [-1.0, 1.0]])
    z1 = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    raw = torch.stack([z0, z1], dim=1)  # [B, T, D]
    path = normalize_lrw_path(raw, z0, z1, requested_n_points=2)
    assert path.shape == (2, 2, 2)
    assert torch.allclose(path[0], z0)
    assert torch.allclose(path[-1], z1)


def test_lrw_bvp_runner_skips_when_lrw_missing(tmp_path, monkeypatch):
    from lrbench.runners.run_lrw_bvp import run

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
                "benchmark_id: lrw_bvp_test_missing",
                "device: cpu",
                "dtype: float32",
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


def test_lrw_bvp_runner_with_fake_lrw_success(tmp_path, monkeypatch):
    from lrbench.runners.run_lrw_bvp import run

    lrw_module = types.ModuleType("lrw")
    geodesic_module = types.ModuleType("lrw.geodesic")
    bvp_module = types.ModuleType("lrw.geodesic.bvp")
    metric_module = types.ModuleType("lrw.metric")
    pullback_module = types.ModuleType("lrw.metric.pullback")

    class FakePullbackMetric:
        def __init__(self, decoder, chunk_size=None, regularization=1e-5):
            self.decoder = decoder
            self.regularization = regularization

        def metric_tensor(self, z):
            return torch.eye(2, device=z.device, dtype=z.dtype).repeat(z.reshape(-1, 2).shape[0], 1, 1)

    class FakeBVPSolver:
        def __init__(self, metric, n_steps=20, step_size=0.05, lr=0.1, max_iter=50, tol=0.001):
            self.metric = metric
            self.n_steps = n_steps

        def geodesic_path(self, z0, z1, n_points=10):
            t = torch.linspace(0, 1, n_points, device=z0.device, dtype=z0.dtype).view(n_points, 1)
            path = (1 - t) * z0 + t * z1
            return path, {"converged": True, "iterations": 1}

    pullback_module.PullbackMetric = FakePullbackMetric
    bvp_module.BVPSolver = FakeBVPSolver
    monkeypatch.setitem(sys.modules, "lrw", lrw_module)
    monkeypatch.setitem(sys.modules, "lrw.geodesic", geodesic_module)
    monkeypatch.setitem(sys.modules, "lrw.geodesic.bvp", bvp_module)
    monkeypatch.setitem(sys.modules, "lrw.metric", metric_module)
    monkeypatch.setitem(sys.modules, "lrw.metric.pullback", pullback_module)

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "\n".join(
            [
                "benchmark_id: lrw_bvp_test_fake",
                "device: cpu",
                "dtype: float32",
                "batch_size: 1",
                "n_points: 6",
                "solver_max_iter: 2",
                "endpoint_tolerance: 0.0001",
                f"output_dir: {tmp_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    result = run(str(cfg))
    assert result["status"] == "success"
    assert result["metrics"]["available"] is True
    assert result["metrics"]["lrw_bvp_endpoint_error_mean"] == 0.0
    assert result["metrics"]["lrw_bvp_num_points_returned"] == 6
