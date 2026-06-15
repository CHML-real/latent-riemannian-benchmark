# LRW Benchmark Status and Attribution

This repository is prepared as a collaborator-facing benchmark suite for the `latent-riemannian-world` / `lrw` package.

## Intended status

The intended status of this project is:

- It is a benchmark suite for latent Riemannian geometry methods.
- It includes adapter-level tests for the LRW package.
- It is suitable to be used as an official LRW benchmark once the LRW maintainer/collaborators are added to the repository or otherwise confirm the repository as the benchmark home.
- It does not vendor, copy, or relicense the LRW package itself. LRW is imported as an installed dependency during adapter tests.

## Ownership boundary

| Item | Ownership / responsibility |
|---|---|
| Benchmark harness | This repository |
| Analytic/synthetic baselines | This repository |
| LRW adapter tests | This repository, for evaluating installed LRW APIs |
| `latent-riemannian-world` / `lrw` package implementation | The LRW project |
| LRW package license | The LRW package's own license |
| This benchmark license | The license file in this repository |

## Recommended public wording

Use wording like:

> This repository provides the LRW collaborator-facing benchmark suite for latent Riemannian geometry APIs. It evaluates analytic baselines, synthetic metrics, pullback-metric correctness, and LRW adapter behavior through reproducible evidence reports.

When collaborator access and repository ownership are finalized, it is reasonable to describe the project as:

> the official LRW benchmark suite

or:

> the official benchmark suite for evaluating LRW adapter behavior and latent Riemannian geometry baselines.

## Interpreting LRW failures

Some LRW adapter benchmarks intentionally report `status=failure`. These are evidence outputs, not harness crashes. A solver-layer failure means that the tested API output did not satisfy the benchmark's endpoint-preserving geodesic expectations under the tested configuration.

The evidence bundle should be read component-wise:

- `PullbackMetric.metric_tensor`: reliable in the tested pullback-metric benchmark.
- `slerp_path`: reliable in the tested spherical interpolation benchmark.
- `GeodesicSolver.interpolate`, `GeodesicSolver.geodesic_distance`, and `BVPSolver.geodesic_path`: require careful semantic interpretation under the benchmark's endpoint-preserving expectations.
