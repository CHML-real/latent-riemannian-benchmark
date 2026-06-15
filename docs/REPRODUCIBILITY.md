# Reproducibility

This document gives a minimal reproducibility recipe for the benchmark and LRW adapter evidence bundle.

## Environment

Use a Python virtual environment. Install the benchmark requirements:

```bash
pip install -r requirements.txt
```

If testing LRW adapter benchmarks, install the LRW package in the same environment. The source probe records the installed file paths and short SHA-256 fingerprints to help identify the implementation being tested.

## Test suite

```bash
python -m pytest -v
```

## Full core benchmark run

```bash
python -m lrbench.runners.run_euclidean --config configs/euclidean_default.yaml
python -m lrbench.runners.run_sphere --config configs/analytic_sphere.yaml
python -m lrbench.runners.run_warped --config configs/synthetic_warped.yaml
python -m lrbench.runners.run_pullback --config configs/synthetic_pullback.yaml
python -m lrbench.runners.run_stress --config configs/stress_scaling.yaml
python -m lrbench.runners.run_report --raw-dir results/raw --report-dir results/reports
```

## Full LRW adapter evidence run

```bash
python -m lrbench.runners.run_lrw_probe --config configs/lrw_probe.yaml
python -m lrbench.runners.run_lrw_introspection --config configs/lrw_api_introspection.yaml
python -m lrbench.runners.run_lrw_pullback_metric --config configs/lrw_pullback_metric.yaml
python -m lrbench.runners.run_lrw_bvp --config configs/lrw_bvp.yaml
python -m lrbench.runners.run_lrw_bvp_diagnostic --config configs/lrw_bvp_diagnostic.yaml
python -m lrbench.runners.run_lrw_euclidean_bvp --config configs/lrw_euclidean_bvp.yaml
python -m lrbench.runners.run_lrw_geodesic_solver --config configs/lrw_geodesic_solver.yaml
python -m lrbench.runners.run_lrw_slerp --config configs/lrw_slerp.yaml
python -m lrbench.runners.run_lrw_geodesic_anatomy --config configs/lrw_geodesic_anatomy.yaml
python -m lrbench.runners.run_lrw_solver_scale_probe --config configs/lrw_solver_scale_probe.yaml
python -m lrbench.runners.run_lrw_source_probe --config configs/lrw_source_probe.yaml
python -m lrbench.runners.run_report --raw-dir results/raw --report-dir results/reports
python -m lrbench.runners.run_lrw_evidence_report --config configs/lrw_evidence_report.yaml
```

## Expected report files

```text
results/raw/*.json
results/reports/summary.md
results/reports/summary.csv
results/reports/lrw_evidence_report.md
results/reports/lrw_evidence_matrix.csv
```

## Windows CMD tips

Use `type`, not PowerShell-only commands:

```bat
type .\results\reports\summary.md
type .\results\reports\lrw_evidence_report.md
python -m json.tool .\results\raw\lrw_source_probe_001.json
```

## Stale result warning

Reports aggregate every JSON file already present in `results/raw`. When upgrading versions, rerun the relevant benchmark commands so old skipped/failure outputs do not appear as current results.
