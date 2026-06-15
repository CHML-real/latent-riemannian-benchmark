from __future__ import annotations

import argparse

from lrbench.reports.summary import generate_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default="results/raw")
    parser.add_argument("--report-dir", type=str, default="results/reports")
    args = parser.parse_args()

    result = generate_report(args.raw_dir, args.report_dir)
    print(f"saved markdown: {result['markdown_path']}")
    print(f"saved csv: {result['csv_path']}")
    print(result)


if __name__ == "__main__":
    main()
