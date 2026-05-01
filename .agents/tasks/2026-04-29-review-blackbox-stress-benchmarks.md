# Review Task: blackbox-stress-benchmarks

## Context
Review the benchmark/metric black-box stress findings produced for the user's SDK stress test. Source-code inspection remains forbidden.

## Goal
Validate `.agents/tmp/blackbox_stress/benchmark_findings.md` for reproducibility, metric interpretation, and fair claim boundaries.

## Responsibility Boundaries
Owns only notes under `.agents/tmp/blackbox_stress/reviews/benchmark_review.md`.

## In Scope
- Read benchmark task doc, generated artifacts, findings, public README/benchmark docs, and CSV/JSON outputs.
- Rerun quick checks or parse artifacts if needed.
- Verify that failures/skips/low metrics are not overstated beyond the evidence contract.

## Out of Scope
- Reading SDK implementation source under `src/**`.
- Modifying benchmark source or promoted benchmark results.
- Fixing bugs.

## Must Not Touch
- `src/**`
- existing benchmark source files
- worker-owned artifacts except by reading/rerunning

## Acceptance Criteria
The review note marks each material benchmark finding as accepted, rejected, or needs-more-evidence, with artifact paths and metric rationale.
