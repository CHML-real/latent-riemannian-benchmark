# Changelog

## v1.1.3 - LRW Benchmark Status Polish

- Clarified repository positioning as an LRW collaborator-facing benchmark suite.
- Added `docs/LRW_BENCHMARK_STATUS.md` for attribution, ownership boundary, and official-benchmark wording.
- Updated citation metadata to describe LRW adapter-level evidence tests.
- Preserved the interpretation that LRW solver-layer failures are benchmark evidence outputs, not harness errors.

All notable changes to this project are documented here.

## [1.1.2] - 2026-06-15

### Added

- GitHub release pack for public distribution.
- `CITATION.cff` for citation metadata.
- `LICENSE` with MIT terms for the benchmark harness.
- GitHub issue templates for benchmark results, LRW adapter evidence, and bug reports.
- `docs/release/RELEASE_NOTES_v1.1.2.md` for the first public release note draft.
- Release-pack documentation tests.

### Clarified

- Expected LRW solver-layer failures are evidence outputs, not harness errors.
- LRW metric/path-function layers and solver layers are reported separately.

## [1.1.1] - 2026-06-15

### Added

- Public release documentation polish.
- `docs/LRW_EVIDENCE_SUMMARY.md`.
- `docs/FAILURE_MODES.md`.
- `docs/REPRODUCIBILITY.md`.

## [1.1.0] - 2026-06-15

### Added

- LRW adapter evidence bundle generator.
- `results/reports/lrw_evidence_report.md`.
- `results/reports/lrw_evidence_matrix.csv`.

## [1.0.x] - 2026-06-15

### Added

- LRW adapter probes for package availability, API introspection, PullbackMetric, BVPSolver, GeodesicSolver, SLERP, output anatomy, scale semantics, and source fingerprinting.

## [0.x] - 2026-06-15

### Added

- Analytic Euclidean baseline.
- Analytic sphere baseline.
- Synthetic warped metric benchmark.
- Synthetic toy decoder pullback benchmark.
- Stress scaling benchmark.
- Global summary report generator.
