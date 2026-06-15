# LRW Adapter Evidence Report

Generated at: `2026-06-15 17:29:28 UTC`

## Executive Summary

This report aggregates the LRW adapter benchmark results into a component-level evidence bundle.

- **Reliable in this benchmark:** `PullbackMetric.metric_tensor`, `slerp_path`, package import/API surface.
- **Risky as endpoint geodesic APIs in this benchmark:** `GeodesicSolver.interpolate`, `GeodesicSolver.geodesic_distance`, `BVPSolver.geodesic_path`.
- **Main diagnosis:** LRW metric/path-function layers pass, but solver-layer endpoint and distance semantics do not match the benchmark's endpoint-preserving geodesic expectations.
- **Key semantic evidence:** `geodesic_distance` scales like `step_size`; `interpolate` source references `shoot` and `step_size`; no tested axis interpretation makes solver outputs endpoint-valid.

## Evidence Counts

- evidence_row_count: `11`
- evidence_pass_count: `7`
- evidence_fail_count: `3`
- evidence_missing_count: `0`
- trusted_component_count: `2`
- risky_component_count: `7`

## Evidence Matrix

| Layer | Component | Benchmark | Status | Key evidence | Interpretation | Failure mode |
|---|---|---|---|---|---|---|
| Adapter availability | `LRW package import` | `lrw_probe_001` | `success` | available=true; public_symbols=6; submodules=16 | The installed LRW package is importable and exposes the expected top-level modules. | `` |
| API surface | `LRW geodesic/metric classes` | `lrw_api_introspection_001` | `success` | classes=4/4; methods=10; signatures=10 | The expected solver, BVP, pullback metric, and base metric APIs are visible for benchmarking. | `` |
| Metric layer | `PullbackMetric.metric_tensor` | `lrw_pullback_metric_001` | `success` | spd_violation=0; ref_rel_frob=0; max_abs=0 | LRW PullbackMetric matches the reference toy-decoder pullback metric and is reliable in this benchmark. | `` |
| Path function layer | `slerp_path` | `lrw_slerp_001` | `success` | endpoint=0; sphere_dist_err=0; radial_max=0; ref_rel=1.879e-08 | The simple spherical interpolation path function preserves endpoints and sphere geometry. | `` |
| Solver layer | `GeodesicSolver.interpolate / geodesic_distance` | `lrw_geodesic_solver_euclidean_001` | `failure` | endpoint=0.068906; path_dist_err=0.956529; geodesic_dist_err=0.99; energy_ratio=0.00188975 | In flat Euclidean sanity testing, GeodesicSolver does not behave as an endpoint-preserving interpolation/distance oracle. | `GEO_F01_ENDPOINT_MISS; DIST_F01_STEP_SIZE_SCALED` |
| BVP solver layer | `BVPSolver.geodesic_path on Euclidean metric` | `lrw_euclidean_bvp_001` | `failure` | endpoint=0.0662638; distance_err=0.361061; energy_ratio=2.13036 | The BVP solver misses endpoint validity even in the flat Euclidean sanity case. | `GEO_F01_ENDPOINT_MISS` |
| BVP solver layer | `BVPSolver.geodesic_path on toy pullback metric` | `lrw_bvp_001` | `failure` | endpoint=1.06321; energy_ratio=0.189056; returned_points=24 | The pullback BVP can reduce energy but cannot be accepted as a valid BVP geodesic when endpoints miss. | `GEO_F01_ENDPOINT_MISS; ENERGY_ONLY_SUCCESS` |
| BVP diagnostics | `BVP best-case and validity scoring` | `lrw_bvp_diagnostic_001` | `success` | case_count=16; raw_endpoint_pass=0; energy_only=5; overall_valid=0; best_endpoint=0.768896 | Energy improvement and endpoint validity are separated; no tested BVP case is overall valid. | `ENERGY_ONLY_SUCCESS; BVP_OVERALL_INVALID` |
| Output anatomy | `Solver output shape semantics` | `lrw_geodesic_output_anatomy_001` | `success` | axis_candidates=2; endpoint_valid_axes=0; best_axis_endpoint=0.0662638; verdict=no_axis_interpretation_pass | The endpoint miss is not explained by a simple [T,B,D] vs [B,T,D] axis interpretation error. | `API_SEM02_NO_AXIS_INTERPRETATION_PASS` |
| Scale semantics | `step_size and n_steps probe` | `lrw_solver_scale_probe_001` | `success` | cases=20; best_endpoint=0.0689055; dist_ratio_range=0.000999999..1; dist_ratio_over_step=1; verdict=no_setting_endpoint_valid;distance_scales_like_step_size | geodesic_distance scales like step_size; no tested step_size/n_steps setting makes interpolate endpoint-valid. | `DIST_F01_STEP_SIZE_SCALED; GEO_F01_ENDPOINT_MISS` |
| Implementation fingerprint | `LRW source/docstring probe` | `lrw_source_probe_001` | `skipped` | targets=0; step_refs=; shoot_refs=; gd_step=; interp_shoot= | The installed source fingerprint supports the experimental reading: distance and interpolate are step/shoot-based rather than plain endpoint interpolation APIs. | `API_SEM01_INTERPOLATE_IS_SHOOT_BASED; DIST_F01_STEP_SIZE_SCALED` |

## Layered Diagnosis

### Metric layer

`PullbackMetric.metric_tensor` matches the reference toy decoder pullback metric with zero relative Frobenius error in the benchmark output. This supports using LRW's metric tensor implementation for metric evaluation.

### Simple path-function layer

`slerp_path` passes endpoint, radial, sphere-distance, and reference-comparison checks. This suggests that LRW's simple spherical interpolation path function is reliable under the tested conditions.

### Solver layer

`GeodesicSolver.interpolate`, `GeodesicSolver.geodesic_distance`, and `BVPSolver.geodesic_path` fail endpoint/distance sanity checks under the benchmark's geodesic API expectations. The failure is not explained by simple tensor-axis misinterpretation, and the scale probe shows distance values scale like `step_size`.

## Failure Modes

| Code | Meaning | Evidence |
|---|---|---|
| `GEO_F01_ENDPOINT_MISS` | Returned path does not satisfy endpoint boundary condition. | Pullback BVP, Euclidean BVP, GeodesicSolver Euclidean, anatomy, and scale probe. |
| `DIST_F01_STEP_SIZE_SCALED` | Distance output scales with solver `step_size` instead of directly matching Euclidean distance. | Scale probe and source probe. |
| `API_SEM01_INTERPOLATE_IS_SHOOT_BASED` | `interpolate` appears to be based on shooting/step rollout semantics rather than guaranteed endpoint interpolation. | Source probe flags and movement-ratio behavior. |
| `API_SEM02_NO_AXIS_INTERPRETATION_PASS` | Endpoint miss is not fixed by common tensor-axis interpretations. | Output anatomy probe. |
| `ENERGY_ONLY_SUCCESS` | Energy can improve while endpoint validity still fails. | BVP diagnostic best-case scoring. |

## Repro Commands

```bat
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
python -m lrbench.runners.run_lrw_evidence_report --config configs/lrw_evidence_report.yaml
```

## Output Files

- `results/reports/lrw_evidence_report.md`
- `results/reports/lrw_evidence_matrix.csv`
