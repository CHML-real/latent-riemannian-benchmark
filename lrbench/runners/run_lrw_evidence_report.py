from __future__ import annotations

import argparse
from typing import Any

from lrbench.reports.lrw_evidence import generate_lrw_evidence_report
from lrbench.utils.io import load_yaml


def run(config_path: str) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    raw_dir = cfg.get("raw_dir", "results/raw")
    report_dir = cfg.get("report_dir", "results/reports")
    result = generate_lrw_evidence_report(raw_dir=raw_dir, report_dir=report_dir)
    print(f"saved markdown: {result['markdown_path']}")
    print(f"saved csv: {result['csv_path']}")
    print(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LRW adapter evidence report")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
