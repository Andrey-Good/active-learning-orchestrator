# 2026-04-28-w08-docs-benchmarks-claims-objection-sweep

## Context

Second-pass exhaustive senior audit requested on 2026-04-28.

## Goal

Audit documentation truthfulness, benchmark claims, benchmark reproducibility, dependency naming, curated/generated result sprawl, and whether README overstates evidence.

## Responsibility Boundaries

Owner may change only:

- `.agents/tmp/2026-04-28-w08-docs-benchmarks-claims-findings.md`

Owner must not change:

- `src/**`
- tests
- existing benchmark scripts/results
- README/docs
- dependency files

## In Scope

- `README.md`
- `benchmarks/*.py`
- `benchmarks/README.md`
- `benchmarks/results/**` at a summary level
- docs audit files for duplicated/conflicting claims

## Out of Scope

- Adding new benchmarks.
- Production code fixes.

## Constraints

- Read-only.
- Do not run long real-dataset benchmarks.
- Findings must cite specific files/lines or artifact values.

## Acceptance Criteria

- Identify overclaims, reproducibility gaps, stale/conflicting docs, and missing caveats.
- Do not create objections merely because a benchmark is limited if the limitation is already clearly stated.

## Dependencies

Can run in parallel with W05, W06, W07, and W09.
