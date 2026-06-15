# Release Notes: v1.1.2 GitHub Release Pack

Date: 2026-06-15

## Summary

This release prepares the Latent Riemannian Benchmark for public GitHub distribution.
It adds citation metadata, license text, changelog entries, issue templates, and a release-note draft while preserving the v1.1.0 LRW evidence-bundle workflow.

## What is included

- Analytic Euclidean and sphere baselines.
- Synthetic warped and toy-decoder pullback metrics.
- Stress scaling benchmark.
- LRW adapter probes for availability, API surface, metric correctness, path functions, solver behavior, output anatomy, scale semantics, and source fingerprinting.
- Evidence report generator for LRW component-level diagnosis.

## Key interpretation

Some LRW adapter benchmarks are expected to report `status=failure`. These are evidence outputs, not harness errors, when the tested solver output does not satisfy endpoint-preserving geodesic expectations.

The current evidence pattern is:

| Component | Interpretation |
|---|---|
| `PullbackMetric.metric_tensor` | Reliable in the tested pullback-metric benchmark. |
| `slerp_path` | Reliable in the tested spherical interpolation benchmark. |
| `GeodesicSolver.interpolate` | Risky as a guaranteed endpoint-preserving interpolation API under this benchmark. |
| `GeodesicSolver.geodesic_distance` | Scales like `step_size` under the Euclidean scale probe. |
| `BVPSolver.geodesic_path` | Fails endpoint validity in Euclidean and toy-pullback BVP checks. |

## Recommended release checklist

Before tagging a public release:

```bash
python -m pytest -v
python -m lrbench.runners.run_report --raw-dir results/raw --report-dir results/reports
python -m lrbench.runners.run_lrw_evidence_report --config configs/lrw_evidence_report.yaml
```

Then verify:

- `results/reports/summary.md`
- `results/reports/summary.csv`
- `results/reports/lrw_evidence_report.md`
- `results/reports/lrw_evidence_matrix.csv`

## Suggested tag

```bash
git tag v1.1.2
git push origin v1.1.2
```
