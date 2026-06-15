from __future__ import annotations

import sys
import types

import torch

from lrbench.methods.lrw_bvp_diagnostic import aggregate_diagnostic_cases, clamp_path_endpoints, summarize_bvp_path


def test_clamp_path_endpoints_forces_boundary_points():
    z0 = torch.tensor([[-2.0, 0.0]])
    z1 = torch.tensor([[2.0, 0.0]])
    path = torch.zeros(4, 1, 2)
    clamped = clamp_path_endpoints(path, z0, z1)
    assert torch.allclose(clamped[0], z0)
    assert torch.allclose(clamped[-1], z1)


def test_summarize_bvp_path_separates_raw_and_clamped_endpoint_errors():
    z0 = torch.tensor([[-2.0, 0.0]])
    z1 = torch.tensor([[2.0, 0.0]])
    path = torch.stack([torch.tensor([[-1.0, 0.0]]), torch.tensor([[0.0, 0.0]]), torch.tensor([[1.0, 0.0]])], dim=0)
    summary = summarize_bvp_path(path, z0, z1, endpoint_tolerance=0.01)
    assert summary["raw_endpoint_pass"] is False
    assert summary["clamped_endpoint_pass"] is True
    assert summary["raw"]["lrw_bvp_endpoint_error_mean"] > 0
    assert summary["clamped"]["clamped_lrw_bvp_endpoint_error_mean"] == 0.0


def test_aggregate_diagnostic_cases_counts_endpoint_failures():
    cases = [
        {"status": "success", "diagnostics": {"raw_endpoint_pass": False, "clamped_endpoint_pass": True, "raw": {"lrw_bvp_endpoint_error_mean": 1.0, "lrw_bvp_pullback_energy_mean": 2.0}, "clamped": {"clamped_lrw_bvp_pullback_energy_mean": 3.0}}},
        {"status": "success", "diagnostics": {"raw_endpoint_pass": True, "clamped_endpoint_pass": True, "raw": {"lrw_bvp_endpoint_error_mean": 0.0, "lrw_bvp_pullback_energy_mean": 1.0}, "clamped": {"clamped_lrw_bvp_pullback_energy_mean": 1.0}}},
    ]
    summary = aggregate_diagnostic_cases(cases)
    assert summary["case_count"] == 2
    assert summary["raw_endpoint_pass_count"] == 1
    assert summary["raw_endpoint_failure_rate"] == 0.5


def test_lrw_bvp_diagnostic_runner_with_fake_lrw(tmp_path, monkeypatch):
    from lrbench.runners.run_lrw_bvp_diagnostic import run

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

        def geodesic_path(self, z0, z1, n_points=10):
            t = torch.linspace(0, 1, n_points, device=z0.device, dtype=z0.dtype).view(n_points, 1, 1)
            path = (1 - t) * z0.unsqueeze(0) + t * z1.unsqueeze(0)
            return path, {"converged": True}

        def solve(self, z0, z1):
            return self.geodesic_path(z0, z1, n_points=4)

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
                "benchmark_id: diag_fake",
                "device: cpu",
                "dtype: float32",
                "methods: [geodesic_path, solve]",
                "n_points: [6]",
                "solver_n_steps: [4]",
                "solver_lr: [0.1]",
                "solver_max_iter: [2]",
                "solver_tol: [0.001]",
                f"output_dir: {tmp_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    result = run(str(cfg))
    assert result["status"] == "success"
    assert result["summary"]["case_count"] == 2
    assert result["summary"]["raw_endpoint_pass_count"] == 2


def test_bvp_best_case_selector_separates_energy_from_validity():
    from lrbench.methods.lrw_bvp_diagnostic import aggregate_diagnostic_cases

    cases = [
        {
            "case_id": "energy_low_endpoint_bad",
            "method_name": "geodesic_path",
            "status": "success",
            "params": {"solver_n_steps": 20},
            "diagnostics": {
                "endpoint_tolerance": 0.01,
                "raw_endpoint_pass": False,
                "clamped_endpoint_pass": True,
                "raw": {
                    "lrw_bvp_endpoint_error_mean": 1.0,
                    "lrw_bvp_pullback_energy_mean": 2.0,
                    "lrw_bvp_pullback_length_mean": 3.0,
                    "lrw_bvp_energy_ratio_over_straight": 0.1,
                },
                "clamped": {"clamped_lrw_bvp_pullback_energy_mean": 9.0},
            },
            "failure": {"failure_type": None},
        },
        {
            "case_id": "endpoint_good_energy_good",
            "method_name": "geodesic_path",
            "status": "success",
            "params": {"solver_n_steps": 20},
            "diagnostics": {
                "endpoint_tolerance": 0.01,
                "raw_endpoint_pass": True,
                "clamped_endpoint_pass": True,
                "raw": {
                    "lrw_bvp_endpoint_error_mean": 0.001,
                    "lrw_bvp_pullback_energy_mean": 5.0,
                    "lrw_bvp_pullback_length_mean": 3.0,
                    "lrw_bvp_energy_ratio_over_straight": 0.8,
                },
                "clamped": {"clamped_lrw_bvp_pullback_energy_mean": 5.0},
            },
            "failure": {"failure_type": None},
        },
    ]
    # Simulate the runner step that attaches compact metrics after diagnostics are computed.
    from lrbench.methods.lrw_bvp_diagnostic import compact_case_metrics
    for case in cases:
        case["compact"] = compact_case_metrics(case)

    summary = aggregate_diagnostic_cases(cases)
    assert summary["energy_improved_count"] == 2
    assert summary["energy_only_success_count"] == 1
    assert summary["overall_valid_count"] == 1
    assert summary["best_by_energy"]["case_id"] == "energy_low_endpoint_bad"
    assert summary["best_overall_valid"]["case_id"] == "endpoint_good_energy_good"
