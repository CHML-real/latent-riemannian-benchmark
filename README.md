# Latent Riemannian Benchmark

A reproducible benchmark suite for evaluating Riemannian geometry computation in latent spaces, including adapter-level benchmarks for the `latent-riemannian-world` / `lrw` package.

This repository is intended to serve as a **collaborator-facing benchmark suite for LRW**. When the LRW maintainer/collaborators adopt or co-maintain this repository, it may be described as an official LRW benchmark suite. Until repository ownership and collaborator access are finalized, the code and reports should be read as an official-benchmark candidate prepared for collaborative use.

The benchmark is designed to separate **metric correctness**, **path-function correctness**, and **solver-layer endpoint/distance semantics**. It currently includes analytic baselines, synthetic warped/pullback metrics, stress scaling, LRW adapter probes, and an evidence-report generator.

> Important: LRW adapter benchmarks may intentionally report failures when endpoint-preserving geodesic expectations are not met. These failures are evidence outputs, not harness errors.

## What this benchmark tests

| Track | Purpose | Example benchmark |
|---|---|---|
| Analytic baselines | Verify basic path/distance correctness on known geometries. | `analytic_euclidean_001`, `analytic_sphere_s2_001` |
| Synthetic metrics | Compare straight paths with geometry-aware detours under controlled metrics. | `synthetic_warped_001`, `synthetic_pullback_001` |
| Stress scaling | Check shape/runtime/stability over dimensions, batches, and path lengths. | `stress_scaling_001` |
| LRW adapter: availability/API | Confirm package import, public API, and signatures. | `lrw_probe_001`, `lrw_api_introspection_001` |
| LRW adapter: metric layer | Compare LRW pullback metric tensors against the reference implementation. | `lrw_pullback_metric_001` |
| LRW adapter: path function | Validate simple spherical interpolation. | `lrw_slerp_001` |
| LRW adapter: solver layer | Test endpoint-preserving geodesic expectations and diagnose mismatches. | `lrw_bvp_001`, `lrw_geodesic_solver_euclidean_001` |
| Evidence bundle | Aggregate pass/fail results into a public report and CSV matrix. | `lrw_evidence_report.md` |

## Quickstart

Install requirements in a Python environment:

```bash
pip install -r requirements.txt
```

Run the test suite:

```bash
python -m pytest -v
```

Run the core non-LRW benchmarks:

```bash
python -m lrbench.runners.run_euclidean --config configs/euclidean_default.yaml
python -m lrbench.runners.run_sphere --config configs/analytic_sphere.yaml
python -m lrbench.runners.run_warped --config configs/synthetic_warped.yaml
python -m lrbench.runners.run_pullback --config configs/synthetic_pullback.yaml
python -m lrbench.runners.run_stress --config configs/stress_scaling.yaml
python -m lrbench.runners.run_report --raw-dir results/raw --report-dir results/reports
```

Run the LRW adapter track:

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

On Windows CMD, inspect reports with:

```bat
type .\results\reports\summary.md
type .\results\reports\lrw_evidence_report.md
```

## Output files

| File | Description |
|---|---|
| `results/raw/*.json` | Per-benchmark machine-readable results. |
| `results/reports/summary.md` | Human-readable global benchmark summary. |
| `results/reports/summary.csv` | Machine-readable global summary. |
| `results/reports/lrw_evidence_report.md` | Component-level LRW adapter evidence report. |
| `results/reports/lrw_evidence_matrix.csv` | LRW evidence matrix for downstream analysis. |

## Interpreting LRW adapter failures

The LRW adapter track intentionally distinguishes package/harness errors from semantic evidence. A failed LRW solver benchmark means the solver output did not satisfy the benchmark's endpoint-preserving geodesic expectations. It does not mean package import failed, and it does not invalidate other LRW components.

Current evidence pattern:

| Component | Benchmark interpretation |
|---|---|
| `PullbackMetric.metric_tensor` | Reliable in this benchmark; matches the reference pullback metric. |
| `slerp_path` | Reliable in this benchmark; preserves endpoints and sphere geometry. |
| `GeodesicSolver.geodesic_distance` | Scales like `step_size` under the tested Euclidean sanity probe; unsafe to treat as raw Euclidean distance without semantic adjustment. |
| `GeodesicSolver.interpolate` | Uses `shoot`/`step_size` semantics in the installed source; unsafe to treat as guaranteed endpoint interpolation. |
| `BVPSolver.geodesic_path` | Fails endpoint validity in both pullback and Euclidean sanity checks under this benchmark. |

See:

- `docs/LRW_EVIDENCE_SUMMARY.md`
- `docs/FAILURE_MODES.md`
- `docs/REPRODUCIBILITY.md`

## Failure modes

The report uses explicit failure-mode codes instead of vague pass/fail labels:

| Code | Meaning |
|---|---|
| `GEO_F01_ENDPOINT_MISS` | Returned path does not satisfy endpoint boundary conditions. |
| `DIST_F01_STEP_SIZE_SCALED` | Distance output scales with solver `step_size` rather than directly matching true Euclidean distance. |
| `API_SEM01_INTERPOLATE_IS_SHOOT_BASED` | `interpolate` appears to be based on shooting/step rollout semantics rather than guaranteed endpoint interpolation. |
| `API_SEM02_NO_AXIS_INTERPRETATION_PASS` | Endpoint miss is not fixed by common tensor-axis interpretations. |
| `ENERGY_ONLY_SUCCESS` | Energy improves while endpoint validity still fails. |

## Project layout

```text
latent-riemannian-benchmark/
├─ README.md
├─ docs/
│  ├─ LRW_EVIDENCE_SUMMARY.md
│  ├─ FAILURE_MODES.md
│  └─ REPRODUCIBILITY.md
├─ configs/
├─ lrbench/
├─ tests/
└─ results/
```

## Scope

This benchmark evaluates latent-space Riemannian geometry computations themselves. It is not tied to a single video generation workflow or downstream application.

## LRW benchmark status and attribution

This project includes a dedicated LRW adapter track for the `latent-riemannian-world` / `lrw` package. The intended positioning is:

- **Benchmark code:** maintained in this repository.
- **LRW package code:** provided by the LRW project and imported as an external dependency during adapter tests.
- **Collaboration status:** prepared for maintainer/collaborator use and suitable to become an official LRW benchmark suite once the LRW maintainer/collaborators are added to the repository.
- **Evidence interpretation:** LRW solver-layer failures are benchmark findings about the tested API semantics, not claims that the entire LRW package is unusable.

The benchmark separates LRW component layers so that reliable components, such as `PullbackMetric.metric_tensor` and `slerp_path` under the tested conditions, can be distinguished from solver-layer endpoint/distance semantics that require care.

## GitHub release pack

This repository includes public-release metadata and issue templates:

| File | Purpose |
|---|---|
| `CHANGELOG.md` | Versioned project history and release notes summary. |
| `CITATION.cff` | Citation metadata for software reuse. |
| `LICENSE` | MIT license for the benchmark harness. |
| `docs/release/RELEASE_NOTES_v1.1.2.md` | Draft release notes for the GitHub release page. |
| `.github/ISSUE_TEMPLATE/benchmark-result.md` | Template for sharing benchmark runs. |
| `.github/ISSUE_TEMPLATE/lrw-adapter-evidence.md` | Template for discussing LRW component-level evidence. |
| `.github/ISSUE_TEMPLATE/bug-report.md` | Template for real harness bugs or crashes. |
| `docs/LRW_BENCHMARK_STATUS.md` | LRW collaborator-facing benchmark status and attribution notes. |

Recommended pre-release check:

```bash
python -m pytest -v
python -m lrbench.runners.run_report --raw-dir results/raw --report-dir results/reports
python -m lrbench.runners.run_lrw_evidence_report --config configs/lrw_evidence_report.yaml
```

When publishing, make clear that this repository is the LRW collaborator-facing benchmark suite and that solver-layer failures such as `GEO_F01_ENDPOINT_MISS` are expected evidence outputs when endpoint-preserving geodesic expectations are not met.
