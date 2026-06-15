# Latent Riemannian Benchmark Summary

Generated at: `2026-06-15 17:29:28 UTC`

## Overall Status

- Total benchmark files: **16**
- Passed: **12**
- Skipped: **1**
- Failed: **3**

## Benchmark Table

| Benchmark | Layer | Manifold | Method | Status | Key metrics |
|---|---|---|---|---|---|
| analytic_euclidean_001 | analytic | euclidean | euclidean_line | success | `endpoint_error_mean=0`<br>`distance_error_mean=4.400e-08`<br>`nan_rate=0`<br>`inf_rate=0`<br>`runtime_ms=5.3177` |
| analytic_sphere_s2_001 | analytic | sphere | sphere_great_circle | success | `endpoint_error_mean=2.931e-08`<br>`sphere_distance_error_mean=6.597e-08`<br>`sphere_radial_error_max=1.192e-07`<br>`nan_rate=0`<br>`inf_rate=0`<br>`runtime_ms=197.239` |
| lrw_api_introspection_001 | adapter | lrw_api | lrw_api_introspection | success | `runtime_ms=1346.95`<br>`api_class_count=4`<br>`api_importable_class_count=4`<br>`api_method_count=10`<br>`api_signature_count=10` |
| lrw_bvp_001 | adapter | toy_decoder_pullback_metric | lrw_bvp_geodesic_path | failure | `lrw_bvp_endpoint_error_mean=1.06321`<br>`lrw_bvp_pullback_energy_mean=51.7681`<br>`lrw_bvp_pullback_length_mean=6.48123`<br>`lrw_bvp_energy_ratio_over_straight=0.189056`<br>`lrw_bvp_num_points_returned=24`<br>`runtime_ms=3656.56` |
| lrw_bvp_diagnostic_001 | adapter | toy_decoder_pullback_metric | lrw_bvp_diagnostic | success | `diagnostic_case_count=16`<br>`diagnostic_call_failure_rate=0.5`<br>`diagnostic_raw_endpoint_failure_rate=1`<br>`diagnostic_raw_endpoint_pass_count=0`<br>`diagnostic_clamped_endpoint_pass_count=8`<br>`diagnostic_raw_endpoint_error_mean_min=0.768896` |
| lrw_euclidean_bvp_001 | adapter | euclidean | lrw_euclidean_bvp_geodesic_path | failure | `lrw_euclidean_bvp_endpoint_error_mean=0.0662638`<br>`lrw_euclidean_bvp_distance_error_mean=0.361061`<br>`lrw_euclidean_bvp_energy_ratio_over_straight=2.13036`<br>`lrw_euclidean_bvp_num_points_returned=24`<br>`runtime_ms=955.493`<br>`available=true` |
| lrw_geodesic_output_anatomy_001 | adapter | euclidean | lrw_geodesic_output_anatomy | success | `anatomy_call_count=4`<br>`anatomy_successful_call_count=3`<br>`anatomy_axis_candidate_count=2`<br>`anatomy_axis_endpoint_valid_count=0`<br>`anatomy_best_axis_endpoint_error=0.0662638`<br>`anatomy_best_axis_distance_error=0.361061` |
| lrw_geodesic_solver_euclidean_001 | adapter | euclidean | lrw_geodesic_solver_euclidean | failure | `lrw_geodesic_solver_endpoint_error_mean=0.068906`<br>`lrw_geodesic_solver_distance_error_mean=0.956529`<br>`lrw_geodesic_solver_geodesic_distance_error_mean=0.99`<br>`lrw_geodesic_solver_energy_ratio_over_straight=0.00188975`<br>`lrw_geodesic_solver_num_points_returned=24`<br>`lrw_geodesic_solver_endpoint_error_mean=0.068906` |
| lrw_probe_001 | adapter | latent_riemannian_world | lrw_package_probe | success | `runtime_ms=1468.93`<br>`available=true`<br>`public_symbol_count=6`<br>`candidate_submodule_count=16` |
| lrw_pullback_metric_001 | adapter | toy_decoder_pullback_metric | lrw_pullback_metric_tensor | success | `lrw_pullback_spd_violation_rate=0`<br>`lrw_vs_reference_relative_frobenius_error_mean=0`<br>`lrw_vs_reference_max_abs_error=0`<br>`sample_count=256`<br>`runtime_ms=1037.74`<br>`available=true` |
| lrw_slerp_001 | adapter | sphere | lrw_slerp_path | success | `lrw_slerp_endpoint_error_mean=0`<br>`lrw_slerp_sphere_distance_error_mean=0`<br>`lrw_slerp_sphere_radial_error_max=0`<br>`lrw_slerp_vs_reference_relative_error_mean=1.879e-08`<br>`lrw_slerp_vs_reference_max_abs_error=5.960e-08`<br>`lrw_slerp_num_points_returned=32` |
| lrw_solver_scale_probe_001 | adapter | euclidean | lrw_solver_scale_probe | success | `scale_probe_case_count=20`<br>`scale_probe_success_count=20`<br>`scale_probe_failure_count=0`<br>`scale_probe_best_endpoint_error=0.0689055`<br>`scale_probe_best_movement_ratio=0.0434782`<br>`scale_probe_distance_ratio_min=0.000999999` |
| lrw_source_probe_001 | adapter | lrw_source | lrw_source_probe | skipped | `source_probe_target_count=0`<br>`source_probe_success_count=0`<br>`source_probe_failure_count=0`<br>`source_probe_target_count=0`<br>`source_probe_success_count=0`<br>`source_probe_failure_count=0` |
| stress_scaling_001 | stress | euclidean_line_scaling | euclidean_line_scaling | success | `runtime_ms_mean=71.1013`<br>`runtime_ms_max=301.507`<br>`failure_rate=0` |
| synthetic_pullback_001 | synthetic | toy_decoder_pullback_metric | straight_vs_pullback_detour | success | `energy_improvement=0.90062`<br>`length_improvement=0.614366`<br>`pullback_spd_violation_rate=0`<br>`runtime_ms=1.67398` |
| synthetic_warped_001 | synthetic | diagonal_warped_metric | straight_vs_warped_detour | success | `energy_improvement=0.853145`<br>`length_improvement=0.648535`<br>`spd_violation_rate=0`<br>`runtime_ms=1.66273` |

## Skipped

- **lrw_source_probe_001**: No module named 'lrw'

## Failures

- **lrw_bvp_001**: `GEO_F01_ENDPOINT_MISS` — LRW BVP path endpoints deviate beyond tolerance
- **lrw_euclidean_bvp_001**: `GEO_F01_ENDPOINT_MISS` — LRW Euclidean BVP path endpoints deviate beyond tolerance
- **lrw_geodesic_solver_euclidean_001**: `GEO_F01_ENDPOINT_MISS` — LRW GeodesicSolver interpolate path endpoints deviate beyond tolerance

## LRW API Introspection

### lrw_api_introspection_001

- Importable classes: `4` / `4`
- Public methods found: `10`
- Signatures found: `10`

| Class | Init signature | Methods |
|---|---|---|
| `lrw.geodesic.bvp.BVPSolver` | `(self, metric: 'RiemannianMetric', n_steps: 'int' = 20, step_size: 'float' = 0.05, lr: 'float' = 0.1, max_iter: 'int' = 50, tol: 'float' = 0.001) -> 'None'` | `geodesic_path(self, z0: 'Tensor', z1: 'Tensor', n_points: 'int' = 10) -> 'tuple[Tensor, dict]'`<br>`solve(self, z0: 'Tensor', z1: 'Tensor') -> 'tuple[Tensor, dict]'` |
| `lrw.geodesic.solver.GeodesicSolver` | `(self, metric: 'RiemannianMetric', n_steps: 'int' = 100, step_size: 'float' = 0.01) -> 'None'` | `geodesic_distance(self, z0: 'Tensor', z1: 'Tensor') -> 'Tensor'`<br>`interpolate(self, z0: 'Tensor', z1: 'Tensor', n_points: 'int' = 10) -> 'Tensor'`<br>`shoot(self, z0: 'Tensor', v0: 'Tensor') -> 'Tensor'` |
| `lrw.metric.pullback.PullbackMetric` | `(self, decoder: 'Callable[[Tensor], Tensor]', chunk_size: 'int | None' = None, regularization: 'float' = 1e-05) -> 'None'` | `local_volume_element(self, z: 'Tensor') -> 'Tensor'`<br>`metric_tensor(self, z: 'Tensor') -> 'Tensor'` |
| `lrw.metric.base.RiemannianMetric` | `(self, /, *args, **kwargs)` | `christoffel(self, z: 'Tensor') -> 'Tensor'`<br>`geodesic_acceleration(self, z: 'Tensor', v: 'Tensor') -> 'Tensor'`<br>`metric_tensor(self, z: 'Tensor') -> 'Tensor'` |

## LRW PullbackMetric Benchmark

### lrw_pullback_metric_001

- Status: `success`
- Sample count: `256`
- LRW SPD violation rate: `0`
- LRW condition number mean: `9.19765`
- Reference same shape: `true`
- Reference relative Frobenius error mean: `0`
- Runtime ms: `1037.74`

## LRW BVPSolver Benchmark

### lrw_bvp_001

- Status: `failure`
- Returned points: `24`
- Endpoint error mean: `1.06321`
- Pullback energy mean: `51.7681`
- Pullback length mean: `6.48123`
- Energy ratio over straight: `0.189056`
- Runtime ms: `3656.56`

## LRW Euclidean BVPSolver Sanity Check

### lrw_euclidean_bvp_001

- Status: `failure`
- Returned points: `24`
- Endpoint error mean: `0.0662638`
- Distance error mean: `0.361061`
- Energy ratio over straight: `2.13036`
- Runtime ms: `955.493`

## LRW GeodesicSolver Euclidean Sanity Check

### lrw_geodesic_solver_euclidean_001

- Status: `failure`
- Returned points: `24`
- Endpoint error mean: `0.068906`
- Path distance error mean: `0.956529`
- geodesic_distance error mean: `0.99`
- Energy ratio over straight: `0.00188975`
- Runtime ms: `42.2341`

## LRW SLERP Sanity Check

### lrw_slerp_001

- Status: `success`
- Function path: `lrw.geodesic.slerp_path`
- Returned points: `32`
- Endpoint error mean: `0`
- Sphere distance error mean: `0`
- Radial error max: `0`
- Reference relative error mean: `1.879e-08`
- Runtime ms: `9.9214`

## LRW Geodesic Solver Output Anatomy

### lrw_geodesic_output_anatomy_001

- Status: `success`
- Call count: `4`
- Successful calls: `3`
- Axis candidate count: `2`
- Axis endpoint-valid count: `0`
- Best axis endpoint error: `0.0662638`
- Best axis: `BVPSolver.geodesic_path` / `as_TBD`
- Best axis distance error: `0.361061`
- Best axis energy ratio: `2.13036`
- geodesic_distance relative error mean: `0.99`
- Verdict: `no_axis_interpretation_pass`
- Runtime ms: `998.741`

| Call | Status | Raw shape | Best candidate | Best endpoint | Best distance error | Energy ratio |
|---|---|---|---|---:|---:|---:|
| `GeodesicSolver.interpolate` | `success` | `[24, 1, 2]` | `as_TBD` | `0.068906` | `0.956529` | `0.00188975` |
| `GeodesicSolver.geodesic_distance` | `success` | `[1]` | `` | `` | `` | `` |
| `BVPSolver.geodesic_path` | `success` | `[24, 1, 2]` | `as_TBD` | `0.0662638` | `0.361061` | `2.13036` |
| `BVPSolver.solve` | `failure` | `` | `` | `` | `` | `` |

## LRW Solver Step-Scale Probe

### lrw_solver_scale_probe_001

- Status: `success`
- Case count: `20`
- Success count: `20`
- Failure count: `0`
- Best endpoint error: `0.0689055`
- Best movement ratio: `0.0434782`
- Distance ratio range: `0.000999999` → `1`
- Movement ratio range: `0.0434711` → `0.0434782`
- Mean distance_ratio / step_size: `1`
- Mean movement_ratio / step_size: `9.83437`
- Verdict: `no_setting_endpoint_valid;distance_scales_like_step_size`
- Runtime ms: `272`

| Case | n_steps | step_size | Status | Distance ratio | Movement ratio | Endpoint error | Ratio/step |
|---|---:|---:|---|---:|---:|---:|---:|
| `case_001` | `1` | `0.001` | `success` | `0.001` | `0.0434782` | `0.0689055` | `1` |
| `case_002` | `1` | `0.01` | `success` | `0.01` | `0.0434782` | `0.0689055` | `1` |
| `case_003` | `1` | `0.05` | `success` | `0.05` | `0.0434782` | `0.0689055` | `1` |
| `case_004` | `1` | `0.1` | `success` | `0.1` | `0.0434782` | `0.0689055` | `1` |
| `case_005` | `1` | `1` | `success` | `1` | `0.0434782` | `0.0689055` | `1` |
| `case_006` | `10` | `0.001` | `success` | `0.001` | `0.0434781` | `0.0689055` | `1` |
| `case_007` | `10` | `0.01` | `success` | `0.01` | `0.0434781` | `0.0689055` | `1` |
| `case_008` | `10` | `0.05` | `success` | `0.05` | `0.0434781` | `0.0689055` | `1` |
| `case_009` | `10` | `0.1` | `success` | `0.1` | `0.0434781` | `0.0689055` | `1` |
| `case_010` | `10` | `1` | `success` | `1` | `0.0434781` | `0.0689055` | `1` |
| `case_011` | `50` | `0.001` | `success` | `0.000999999` | `0.0434782` | `0.0689055` | `0.999999` |
| `case_012` | `50` | `0.01` | `success` | `0.01` | `0.0434782` | `0.0689055` | `1` |
| `case_013` | `50` | `0.05` | `success` | `0.05` | `0.0434782` | `0.0689055` | `1` |
| `case_014` | `50` | `0.1` | `success` | `0.1` | `0.0434782` | `0.0689055` | `1` |
| `case_015` | `50` | `1` | `success` | `0.999999` | `0.0434782` | `0.0689055` | `0.999999` |
| `case_016` | `100` | `0.001` | `success` | `0.001` | `0.0434711` | `0.068906` | `1` |
| `case_017` | `100` | `0.01` | `success` | `0.01` | `0.0434711` | `0.068906` | `1` |
| `case_018` | `100` | `0.05` | `success` | `0.05` | `0.0434711` | `0.068906` | `0.999999` |
| `case_019` | `100` | `0.1` | `success` | `0.0999999` | `0.0434711` | `0.068906` | `0.999999` |
| `case_020` | `100` | `1` | `success` | `0.999999` | `0.0434711` | `0.068906` | `0.999999` |

## LRW Source / Implementation Fingerprint Probe

### lrw_source_probe_001

- Status: `skipped`
- Target count: `0`
- Success count: `0`
- Failure count: `0`
- step_size references: ``
- shoot references: ``
- num_points references: ``
- geodesic_distance contains step_size: ``
- interpolate contains shoot: ``
- interpolate contains step_size: ``
- BVPSolver.geodesic_path contains solve: ``
- BVPSolver.solve contains num_points: ``
- Verdict: `lrw_unavailable`
- Runtime ms: `0.521868`
- Skip reason: No module named 'lrw'

## LRW BVP Diagnostic

### lrw_bvp_diagnostic_001

- Status: `success`
- Case count: `16`
- Call failure rate: `0.5`
- Raw endpoint failure rate: `1`
- Raw endpoint pass count: `0`
- Clamped endpoint pass count: `8`
- Raw endpoint error range: `0.768896` → `43.2778`
- Energy-improved cases: `5`
- Energy-only successes: `5`
- Overall valid cases: `0`
- Overall valid rate: `0`
- Best endpoint case: `case_008` endpoint=`0.768896` energy_ratio=`0.256915`
- Best energy case: `case_001` endpoint=`1.15165` energy_ratio=`0.139805`
- Best validity-score case: `case_008` score=`0.00521591` overall_valid=`false`
- Runtime ms: `1.401e+05`

| Case | Method | Params | Status | Raw endpoint | Energy ratio | Overall valid | Valid score |
|---|---|---|---|---:|---:|---:|---:|
| `case_001` | `geodesic_path` | `n_steps=20, lr=0.05, max_iter=30` | `success` | `1.15165` | `0.139805` | `false` | `0.00386957` |
| `case_002` | `geodesic_path` | `n_steps=20, lr=0.05, max_iter=100` | `success` | `1.038` | `0.194019` | `false` | `0.00409631` |
| `case_003` | `geodesic_path` | `n_steps=20, lr=0.1, max_iter=30` | `success` | `1.06321` | `0.189056` | `false` | `0.00401705` |
| `case_004` | `geodesic_path` | `n_steps=20, lr=0.1, max_iter=100` | `success` | `1.03799` | `0.19411` | `false` | `0.00409598` |
| `case_005` | `geodesic_path` | `n_steps=40, lr=0.05, max_iter=30` | `success` | `36.9711` | `430.186` | `false` | `3.381e-05` |
| `case_006` | `geodesic_path` | `n_steps=40, lr=0.05, max_iter=100` | `success` | `43.2778` | `596.157` | `false` | `2.888e-05` |
| `case_007` | `geodesic_path` | `n_steps=40, lr=0.1, max_iter=30` | `success` | `10.4611` | `45.6544` | `false` | `0.000119433` |
| `case_008` | `geodesic_path` | `n_steps=40, lr=0.1, max_iter=100` | `success` | `0.768896` | `0.256915` | `false` | `0.00521591` |

## Stress Summary

### stress_scaling_001

- Case count: `24`
- Failure rate: `0`
- Runtime mean ms: `71.1013`
- Runtime max ms: `301.507`
- Max distance error: `1.207e-07`

## CSV Columns

A machine-readable CSV version is written next to this Markdown report.

```txt
benchmark_id, benchmark_layer, manifold, method, status, device, dtype, dimension, batch_size, num_points, case_count, failure_count, failure_rate, endpoint_error_mean, distance_error_mean, sphere_distance_error_mean, energy_improvement, length_improvement, spd_violation_rate, pullback_spd_violation_rate, lrw_pullback_spd_violation_rate, lrw_vs_reference_relative_frobenius_error_mean, lrw_vs_reference_max_abs_error, sample_count, lrw_bvp_endpoint_error_mean, lrw_bvp_pullback_energy_mean, lrw_bvp_pullback_length_mean, lrw_bvp_energy_ratio_over_straight, lrw_bvp_num_points_returned, lrw_euclidean_bvp_endpoint_error_mean, lrw_euclidean_bvp_distance_error_mean, lrw_euclidean_bvp_energy_ratio_over_straight, lrw_euclidean_bvp_num_points_returned, lrw_geodesic_solver_endpoint_error_mean, lrw_geodesic_solver_distance_error_mean, lrw_geodesic_solver_geodesic_distance_error_mean, lrw_geodesic_solver_energy_ratio_over_straight, lrw_geodesic_solver_num_points_returned, lrw_slerp_endpoint_error_mean, lrw_slerp_sphere_distance_error_mean, lrw_slerp_sphere_radial_error_max, lrw_slerp_vs_reference_relative_error_mean, lrw_slerp_vs_reference_max_abs_error, lrw_slerp_num_points_returned, anatomy_call_count, anatomy_successful_call_count, anatomy_axis_candidate_count, anatomy_axis_endpoint_valid_count, anatomy_best_axis_endpoint_error, anatomy_best_axis_distance_error, anatomy_best_axis_energy_ratio, anatomy_geodesic_distance_relative_error_mean, scale_probe_case_count, scale_probe_success_count, scale_probe_failure_count, scale_probe_best_endpoint_error, scale_probe_best_movement_ratio, scale_probe_distance_ratio_min, scale_probe_distance_ratio_max, scale_probe_movement_ratio_min, scale_probe_movement_ratio_max, scale_probe_distance_ratio_over_step_size_mean, scale_probe_movement_ratio_over_step_size_mean, diagnostic_case_count, diagnostic_call_failure_rate, diagnostic_raw_endpoint_failure_rate, diagnostic_raw_endpoint_pass_count, diagnostic_clamped_endpoint_pass_count, diagnostic_raw_endpoint_error_mean_min, diagnostic_raw_endpoint_error_mean_max, diagnostic_energy_improved_count, diagnostic_energy_only_success_count, diagnostic_overall_valid_count, diagnostic_overall_valid_rate, diagnostic_endpoint_tolerance_pass_rate, diagnostic_boundary_condition_failure_count, diagnostic_best_raw_endpoint_error, diagnostic_best_raw_energy_ratio, diagnostic_best_valid_score, nan_rate, inf_rate, runtime_ms, runtime_ms_mean, runtime_ms_max, available, module_name, version, api_class_count, api_importable_class_count, api_method_count, api_signature_count, sample_count, regularization, lrw_pullback_spd_violation_rate, lrw_pullback_condition_number_mean, lrw_pullback_condition_number_max, lrw_vs_reference_same_shape, lrw_vs_reference_mean_abs_error, lrw_vs_reference_max_abs_error, lrw_vs_reference_relative_frobenius_error_mean, lrw_vs_reference_relative_frobenius_error_max
```
