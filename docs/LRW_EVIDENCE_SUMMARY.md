# LRW Evidence Summary

This document summarizes the LRW adapter evidence track in a public-release format.

## Executive conclusion

The benchmark separates LRW behavior into layers rather than treating the package as a single pass/fail target.

| Layer | Component | Evidence status | Interpretation |
|---|---|---|---|
| Adapter availability | Package import/API surface | Pass | LRW is importable and exposes the expected public modules/classes. |
| Metric layer | `PullbackMetric.metric_tensor` | Pass | Metric tensors match the benchmark reference pullback metric. |
| Simple path-function layer | `slerp_path` | Pass | Spherical interpolation preserves endpoints and sphere geometry. |
| Solver layer | `GeodesicSolver.interpolate` | Risk/fail under endpoint-preserving interpretation | Output does not reach the requested endpoint in Euclidean sanity checks. |
| Solver layer | `GeodesicSolver.geodesic_distance` | Risk/fail as raw distance oracle | Distance scales like `step_size` in the scale probe. |
| BVP solver layer | `BVPSolver.geodesic_path` | Fail under endpoint-preserving BVP expectations | Endpoint validity fails in Euclidean and toy pullback checks. |

## Main diagnosis

LRW metric/path-function layers pass the benchmark, but solver-layer endpoint and distance semantics do not match the benchmark's endpoint-preserving geodesic expectations.

The evidence does not say that LRW is globally unusable. It says that, in this benchmark, only some LRW components are safe to use as direct reference implementations for endpoint-preserving latent geodesic evaluation.

## Strong evidence points

1. `PullbackMetric.metric_tensor` matches the reference metric with zero relative Frobenius error on the toy decoder pullback benchmark.
2. `slerp_path` passes endpoint, radial, spherical distance, and reference-comparison checks.
3. `GeodesicSolver.interpolate` misses the target endpoint under a flat Euclidean sanity check.
4. `GeodesicSolver.geodesic_distance` returns values proportional to `step_size` under the step-scale probe.
5. Output anatomy rejects common axis interpretation mistakes as the cause of endpoint miss.
6. Source fingerprinting shows `interpolate` references `shoot`/`step_size` and `geodesic_distance` references `step_size`, matching the numeric behavior.

## Suggested public wording

> This benchmark distinguishes reliable LRW components from solver-layer semantic risks. LRW's `PullbackMetric.metric_tensor` and `slerp_path` pass the tested reference checks. However, `GeodesicSolver.interpolate`, `GeodesicSolver.geodesic_distance`, and `BVPSolver.geodesic_path` do not satisfy the benchmark's endpoint-preserving geodesic expectations under Euclidean and toy pullback sanity tests. These failures are benchmark evidence outputs, not harness crashes.

## Related generated files

After running the full LRW adapter track, inspect:

```text
results/reports/summary.md
results/reports/summary.csv
results/reports/lrw_evidence_report.md
results/reports/lrw_evidence_matrix.csv
```
