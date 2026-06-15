from __future__ import annotations

from lrbench.metrics.scaling import group_by_batch_size, group_by_dimension, summarize_stress_cases
from lrbench.runners.run_stress import _run_case
from lrbench.utils.device import resolve_device, resolve_dtype


def test_stress_single_case_success():
    result = _run_case(
        benchmark_id="test_stress",
        seed=123,
        device=resolve_device("cpu"),
        dtype=resolve_dtype("float32"),
        dimension=8,
        batch_size=4,
        num_points=16,
        max_endpoint_error=1e-5,
        max_distance_error=1e-5,
    )
    assert result["status"] == "success"
    assert result["metrics"]["endpoint_error_mean"] < 1e-6
    assert result["metrics"]["distance_error_mean"] < 1e-5
    assert result["metrics"]["has_nan"] is False
    assert result["metrics"]["has_inf"] is False


def test_scaling_summary_counts_failures():
    cases = [
        {"status": "success", "dimension": 2, "batch_size": 1, "metrics": {"runtime_ms": 1.0, "peak_memory_mb": 0.0, "nan_rate": 0.0, "inf_rate": 0.0, "endpoint_error_mean": 0.0, "distance_error_mean": 0.0}},
        {"status": "failure", "dimension": 8, "batch_size": 1, "metrics": {"runtime_ms": 2.0, "peak_memory_mb": 0.0, "nan_rate": 1.0, "inf_rate": 0.0, "endpoint_error_mean": 1.0, "distance_error_mean": 1.0}},
    ]
    summary = summarize_stress_cases(cases)
    assert summary["case_count"] == 2
    assert summary["success_count"] == 1
    assert summary["failure_count"] == 1
    assert summary["failure_rate"] == 0.5


def test_scaling_grouping_returns_dimension_and_batch_tables():
    cases = [
        {"status": "success", "dimension": 2, "batch_size": 1, "metrics": {"runtime_ms": 1.0, "peak_memory_mb": 0.0, "nan_rate": 0.0, "inf_rate": 0.0, "endpoint_error_mean": 0.0, "distance_error_mean": 0.0}},
        {"status": "success", "dimension": 2, "batch_size": 4, "metrics": {"runtime_ms": 2.0, "peak_memory_mb": 0.0, "nan_rate": 0.0, "inf_rate": 0.0, "endpoint_error_mean": 0.0, "distance_error_mean": 0.0}},
    ]
    by_dim = group_by_dimension(cases)
    by_batch = group_by_batch_size(cases)
    assert by_dim["2"]["case_count"] == 2
    assert by_batch["1"]["case_count"] == 1
    assert by_batch["4"]["case_count"] == 1
