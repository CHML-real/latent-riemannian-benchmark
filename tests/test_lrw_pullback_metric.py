from __future__ import annotations

import sys
import types

import torch

from lrbench.methods.lrw_pullback_metric import (
    compare_metric_tensors,
    compute_reference_pullback_metric,
    metric_tensor_stats,
)


def test_metric_tensor_stats_detects_spd():
    G = torch.eye(2).repeat(5, 1, 1)
    stats = metric_tensor_stats(G, prefix="test")
    assert stats["test_shape"] == [5, 2, 2]
    assert stats["test_spd_violation_rate"] == 0.0
    assert stats["test_has_nan"] is False
    assert stats["test_has_inf"] is False


def test_compare_metric_tensors_identical_is_zero():
    G = torch.eye(2).repeat(3, 1, 1)
    metrics = compare_metric_tensors(G, G.clone())
    assert metrics["lrw_vs_reference_same_shape"] is True
    assert metrics["lrw_vs_reference_mean_abs_error"] == 0.0
    assert metrics["lrw_vs_reference_relative_frobenius_error_mean"] == 0.0


def test_reference_pullback_metric_shape_and_spd():
    z = torch.tensor([[-2.0, 0.0], [0.0, 0.0], [2.0, 1.5]], dtype=torch.float32)
    G = compute_reference_pullback_metric(z)
    assert G.shape == (3, 2, 2)
    eigvals = torch.linalg.eigvalsh(G)
    assert torch.all(eigvals > 0)


def test_lrw_pullback_metric_runner_skips_when_lrw_missing(tmp_path, monkeypatch):
    from lrbench.runners.run_lrw_pullback_metric import run

    # Force import attempts for LRW to fail, regardless of local environment.
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
                "benchmark_id: lrw_pullback_metric_test_missing",
                "device: cpu",
                "dtype: float32",
                "batch_size: 2",
                "num_points: 4",
                "max_metric_points: 8",
                f"output_dir: {tmp_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    result = run(str(cfg))
    assert result["status"] == "skipped"
    assert result["skip"]["is_skipped"] is True


def test_lrw_pullback_metric_runner_with_fake_lrw_success(tmp_path, monkeypatch):
    # Build fake module hierarchy: lrw.metric.pullback.PullbackMetric
    from lrbench.runners.run_lrw_pullback_metric import run

    lrw_module = types.ModuleType("lrw")
    metric_module = types.ModuleType("lrw.metric")
    pullback_module = types.ModuleType("lrw.metric.pullback")

    class FakePullbackMetric:
        def __init__(self, decoder, chunk_size=None, regularization=1e-5):
            self.decoder = decoder
            self.regularization = regularization

        def metric_tensor(self, z):
            from lrbench.manifolds.pullback import toy_pullback_metric

            return toy_pullback_metric(z, damping=self.regularization)

    pullback_module.PullbackMetric = FakePullbackMetric
    monkeypatch.setitem(sys.modules, "lrw", lrw_module)
    monkeypatch.setitem(sys.modules, "lrw.metric", metric_module)
    monkeypatch.setitem(sys.modules, "lrw.metric.pullback", pullback_module)

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "\n".join(
            [
                "benchmark_id: lrw_pullback_metric_test_fake",
                "device: cpu",
                "dtype: float32",
                "batch_size: 2",
                "num_points: 4",
                "max_metric_points: 8",
                "regularization: 0.00001",
                f"output_dir: {tmp_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    result = run(str(cfg))
    assert result["status"] == "success"
    assert result["metrics"]["available"] is True
    assert result["metrics"]["lrw_pullback_spd_violation_rate"] == 0.0
    assert result["metrics"]["lrw_vs_reference_same_shape"] is True
