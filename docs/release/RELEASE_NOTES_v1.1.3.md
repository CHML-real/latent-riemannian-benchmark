# Release Notes: v1.1.3 LRW Benchmark Status Polish

Date: 2026-06-15

## Summary

This release clarifies the repository positioning for LRW collaboration. The project is now documented as a collaborator-facing benchmark suite for `latent-riemannian-world` / `lrw`, suitable to become or be described as the official LRW benchmark once maintainer/collaborator access is finalized.

## What changed

- Clarified README wording around LRW benchmark status.
- Added `docs/LRW_BENCHMARK_STATUS.md`.
- Updated citation metadata to describe LRW adapter-level evidence tests.
- Preserved the evidence interpretation that solver-layer failures are benchmark findings, not harness errors.

## Recommended description

> A reproducible LRW collaborator-facing benchmark suite for latent Riemannian geometry APIs, including analytic baselines, synthetic metric tests, LRW adapter probes, and component-level evidence reports.

## Important interpretation

Some LRW adapter benchmarks are expected to report `status=failure`. These failures are evidence outputs when the tested solver output does not satisfy endpoint-preserving geodesic expectations. They should not be interpreted as package import failures or benchmark harness crashes.
