# Failure Modes

This document defines the benchmark's failure-mode codes. The intent is to make solver behavior diagnosable rather than reducing every result to a vague pass/fail label.

## Codes

| Code | Meaning | Typical evidence |
|---|---|---|
| `GEO_F01_ENDPOINT_MISS` | A returned path does not satisfy the requested start/end boundary condition. | Endpoint error exceeds tolerance. |
| `DIST_F01_STEP_SIZE_SCALED` | A distance value scales with solver `step_size` rather than directly matching the analytic distance. | `geodesic_distance / true_distance ≈ step_size`. |
| `API_SEM01_INTERPOLATE_IS_SHOOT_BASED` | `interpolate` appears to use shooting or step rollout semantics rather than guaranteed endpoint interpolation. | Source fingerprint flags `shoot` and `step_size`; movement ratio does not reach target. |
| `API_SEM02_NO_AXIS_INTERPRETATION_PASS` | Common tensor-axis interpretations do not fix endpoint miss. | Anatomy probe finds zero endpoint-valid axis candidates. |
| `ENERGY_ONLY_SUCCESS` | Energy improves while endpoint validity fails. | Energy ratio below baseline but endpoint error exceeds tolerance. |
| `BVP_OVERALL_INVALID` | BVP call returns a path but fails the combined validity criteria. | Endpoint validity count is zero even with energy-improved cases. |

## Why failures are useful

The LRW adapter track intentionally reports some solver-layer failures. These failures are part of the evidence bundle and help separate:

- import/API availability,
- metric tensor correctness,
- simple path-function correctness,
- endpoint-preserving solver semantics,
- and step-size-scaled solver behavior.

A failure in `lrw_bvp_001`, for example, does not imply that `PullbackMetric.metric_tensor` is wrong. It means that the returned BVP path did not satisfy the benchmark's endpoint validity criterion.

## Recommended interpretation

Use the failure modes as diagnostic labels:

```text
Metric layer PASS + Solver layer FAIL
```

means the package can provide useful metric tensors while still being risky as a direct endpoint-preserving geodesic solver under the benchmark's expectations.
