from __future__ import annotations

import argparse
import time
from typing import Any

from lrbench.methods.lrw_introspection import introspect_lrw_api
from lrbench.utils.io import load_yaml, save_json


def run_lrw_introspection(config_path: str) -> dict[str, Any]:
    config = load_yaml(config_path)
    start = time.perf_counter()
    api = introspect_lrw_api(config.get("class_paths"))
    runtime_ms = (time.perf_counter() - start) * 1000.0

    all_importable = bool(api.get("all_importable"))
    importable_count = int(api.get("importable_class_count", 0))
    class_count = int(api.get("class_count", 0))
    status = "success" if all_importable else ("skipped" if importable_count == 0 else "failure")

    result: dict[str, Any] = {
        "benchmark_id": config.get("benchmark_id", "lrw_api_introspection_001"),
        "benchmark_layer": config.get("benchmark_layer", "adapter"),
        "target": config.get("target", "lrw_api"),
        "method": config.get("method", "lrw_api_introspection"),
        "status": status,
        "api": api,
        "metrics": {
            "api_class_count": class_count,
            "api_importable_class_count": importable_count,
            "api_method_count": api.get("method_count", 0),
            "api_signature_count": api.get("signature_count", 0),
            "runtime_ms": runtime_ms,
        },
        "failure": {
            "has_failure": status == "failure",
            "failure_type": None if status != "failure" else "LRW_API_PARTIAL_IMPORT_FAILURE",
            "message": None if status != "failure" else f"Only {importable_count}/{class_count} LRW classes were importable.",
        },
        "skip": {
            "is_skipped": status == "skipped",
            "reason": None if status != "skipped" else "No LRW API classes were importable.",
        },
    }

    output_path = config.get("output_path", "results/raw/lrw_api_introspection_001.json")
    save_json(result, output_path)
    print(f"saved: {output_path}")
    print(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lrw_api_introspection.yaml")
    args = parser.parse_args()
    run_lrw_introspection(args.config)


if __name__ == "__main__":
    main()
