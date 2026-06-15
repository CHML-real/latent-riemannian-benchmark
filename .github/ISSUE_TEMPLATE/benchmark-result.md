---
name: Benchmark result report
about: Share a benchmark run, result summary, or reproducibility issue
title: "[Benchmark Result] "
labels: benchmark-result
assignees: ''
---

## Environment

- OS:
- Python version:
- PyTorch version:
- LRW version, if used:
- Device:

## Commands run

```bash
python -m pytest -v
python -m lrbench.runners.run_report --raw-dir results/raw --report-dir results/reports
```

## Summary

Paste the top of `results/reports/summary.md` here:

```text

```

## Notes

Expected LRW solver-layer failures should be reported as evidence outputs, not harness errors, when endpoint-preserving geodesic expectations are not met.
