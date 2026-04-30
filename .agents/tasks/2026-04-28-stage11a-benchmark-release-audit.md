# Stage 11A: Real-Data Benchmark Release Audit

## Task Identifier

stage11a-benchmark-release-audit

## Context

Stage 11 roadmap goal is to make benchmark evidence reproducible and useful on
real datasets, not only synthetic diagnostics. Existing code already has real
dataset presets and many retained artifacts, but we need a senior audit before
implementation.

## Goal

Audit benchmark harness readiness for Stage 11 and identify P1/P2 blockers.

## Responsibility Boundaries

In scope:

- `benchmarks/sdk_first_benchmark.py`
- `benchmarks/quality_gate_report.py`
- benchmark-related tests
- `benchmarks/README.md`
- `docs/BENCHMARK_EVIDENCE.md`
- README benchmark claims

Out of scope:

- Editing production SDK code.
- Running long real-data benchmarks.
- Changing retained historical artifacts.

## Files May Be Changed

- Only `.agents/tmp/2026-04-28-stage11a-benchmark-release-audit.md`

## Review Questions

1. Does the harness support real datasets with explicit caps and reproducibility
   metadata?
2. Are at least three seeds encouraged or enforced for standard real reports?
3. Are calibration metrics (`ECE`, `Brier`, `NLL`) computed and reported?
4. Are runtime, AULC, lift-vs-random, coverage, rare/zero-recall, and stop-policy
   metrics present?
5. Are direct external-library workflow claims separated from formula shims?
6. Are docs honest about capped-real evidence and limitations?
7. Are tests sufficient to prevent evidence/claim drift?

## Expected Output

Write a severity-ranked report with:

- verdict: accept/reject for Stage 11 readiness;
- P1/P2 blockers with concrete evidence;
- P3/future work separated from blockers;
- suggested fix boundaries.

## Forbidden Actions

- Do not edit benchmark code/tests/docs.
- Do not invent benchmark results.
- Do not download datasets.

## Acceptance Criteria

- Report is actionable enough for a worker task.
- Blockers are tied to Stage 11 exit criteria, not generic wishlist items.
