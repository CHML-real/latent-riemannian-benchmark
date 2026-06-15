from __future__ import annotations

from typing import Iterable


def summarize_stress_cases(cases: list[dict]) -> dict:
    if not cases:
        return {
            "case_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "failure_rate": 1.0,
        }

    success_count = sum(1 for c in cases if c.get("status") == "success")
    failure_count = len(cases) - success_count
    runtimes = [float(c["metrics"]["runtime_ms"]) for c in cases]
    memories = [float(c["metrics"].get("peak_memory_mb", 0.0)) for c in cases]
    nan_rates = [float(c["metrics"].get("nan_rate", 0.0)) for c in cases]
    inf_rates = [float(c["metrics"].get("inf_rate", 0.0)) for c in cases]
    endpoint_errors = [float(c["metrics"].get("endpoint_error_mean", 0.0)) for c in cases]
    distance_errors = [float(c["metrics"].get("distance_error_mean", 0.0)) for c in cases]

    return {
        "case_count": len(cases),
        "success_count": success_count,
        "failure_count": failure_count,
        "failure_rate": failure_count / len(cases),
        "runtime_ms_mean": sum(runtimes) / len(runtimes),
        "runtime_ms_max": max(runtimes),
        "peak_memory_mb_max": max(memories),
        "nan_rate_max": max(nan_rates),
        "inf_rate_max": max(inf_rates),
        "endpoint_error_mean_max": max(endpoint_errors),
        "distance_error_mean_max": max(distance_errors),
    }


def group_by_dimension(cases: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for case in cases:
        key = str(case.get("dimension"))
        groups.setdefault(key, []).append(case)
    return {key: summarize_stress_cases(value) for key, value in groups.items()}


def group_by_batch_size(cases: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for case in cases:
        key = str(case.get("batch_size"))
        groups.setdefault(key, []).append(case)
    return {key: summarize_stress_cases(value) for key, value in groups.items()}
