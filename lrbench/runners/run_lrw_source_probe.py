from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from lrbench.methods.lrw_source_probe import run_lrw_source_probe
from lrbench.utils.io import load_yaml, save_json


def run(config_path: str) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    max_source_lines = int(cfg.get("max_source_lines", 80))

    start = time.perf_counter()
    entries, lrw_info, summary = run_lrw_source_probe(max_source_lines=max_source_lines)
    runtime_ms = (time.perf_counter() - start) * 1000.0

    metrics: dict[str, Any] = {**summary, "available": bool(lrw_info.get("available", False)), "runtime_ms": runtime_ms, "peak_memory_mb": 0.0}
    status = "success"
    skip = {"is_skipped": False, "reason": None}
    failure = {"has_failure": False, "failure_type": None, "message": None}
    if not lrw_info.get("available", False):
        status = "skipped"
        skip = {"is_skipped": True, "reason": lrw_info.get("reason", "LRW unavailable")}
    elif summary.get("source_probe_success_count", 0) == 0:
        status = "failure"
        failure = {"has_failure": True, "failure_type": "GEO_F09_SOURCE_PROBE_EMPTY", "message": "No LRW source targets could be inspected"}

    result = {
        "benchmark_id": cfg.get("benchmark_id", "lrw_source_probe_001"),
        "benchmark_layer": "adapter",
        "target": "lrw_source_probe",
        "manifold": "lrw_source",
        "method": "lrw_source_probe",
        "status": status,
        "max_source_lines": max_source_lines,
        "lrw": lrw_info,
        "summary": summary,
        "metrics": metrics,
        "source_entries": entries,
        "failure": failure,
        "skip": skip,
    }

    output_dir = Path(cfg.get("output_dir", "results/raw"))
    output_path = output_dir / f"{result['benchmark_id']}.json"
    save_json(result, output_path)
    print(f"saved: {output_path}")
    print(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LRW source/docstring fingerprint probe")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
