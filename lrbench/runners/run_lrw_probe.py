from __future__ import annotations

import argparse
import time
from typing import Any

from lrbench.methods.lrw_adapter import probe_lrw_package
from lrbench.utils.io import load_yaml, save_json


def run_lrw_probe(config_path: str) -> dict[str, Any]:
    config = load_yaml(config_path)
    start = time.perf_counter()
    probe = probe_lrw_package(config.get("module_candidates"))
    runtime_ms = (time.perf_counter() - start) * 1000.0

    available = bool(probe.get("available"))
    status = "success" if available else "skipped"

    result: dict[str, Any] = {
        "benchmark_id": config.get("benchmark_id", "lrw_probe_001"),
        "benchmark_layer": config.get("benchmark_layer", "adapter"),
        "target": config.get("target", "latent_riemannian_world"),
        "method": config.get("method", "lrw_package_probe"),
        "status": status,
        "available": available,
        "module_name": probe.get("module_name"),
        "version": probe.get("version"),
        "module_file": probe.get("module_file"),
        "public_symbols": probe.get("public_symbols", []),
        "candidate_submodules": probe.get("candidate_submodules", []),
        "metrics": {
            "available": available,
            "public_symbol_count": len(probe.get("public_symbols", [])),
            "candidate_submodule_count": len(probe.get("candidate_submodules", [])),
            "runtime_ms": runtime_ms,
        },
        "failure": {
            "has_failure": False,
            "failure_type": None,
            "message": None,
        },
        "skip": {
            "is_skipped": not available,
            "reason": None if available else probe.get("error", "LRW package is not installed."),
        },
    }

    output_path = config.get("output_path", "results/raw/lrw_probe_001.json")
    save_json(result, output_path)
    print(f"saved: {output_path}")
    print(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lrw_probe.yaml")
    args = parser.parse_args()
    run_lrw_probe(args.config)


if __name__ == "__main__":
    main()
