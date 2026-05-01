# Task: wave5-quality-and-benchmark

## Context
Repeat black-box benchmark stress after another claimed fix round. Prior Wave4 retained real quality passed but adversarial tiny synthetic differentiation failed.

## Goal
Re-test Wave4 quality findings and search for new benchmark/metric regressions.

## Responsibility Boundaries
Owns only `.agents/tmp/blackbox_stress/wave5_quality/` and `.agents/tmp/blackbox_stress/wave5_quality_findings.md`.

## In Scope
- Read public docs and prior black-box reports.
- Use documented benchmark CLIs only; do not inspect benchmark implementation source.
- Re-test:
  - adversarial tiny-budget synthetic differentiation;
  - retained Banking77 and Emotion gates, preferably using prior commands or bounded variants;
  - budget warnings below seed size.
- Search new benchmark issues: missing artifacts, manifest/category inconsistencies, quality_gate parser behavior, fallback/aliasing diagnostics, calibration columns, runtime regressions, seed count/cap evidence.

## Out of Scope
- Reading SDK source or benchmark implementation source.
- Modifying benchmark source/results outside owned output dir.
- Broad uncapped claims.

## Must Not Touch
- `src/**`
- benchmark source files
- promoted result dirs
- sibling wave5 directories

## Acceptance Criteria
- At least 3 benchmark commands attempted, unless dependency/network failure blocks them.
- At least 4 strategy families covered.
- Explicitly state whether Wave4 synthetic differentiation failure is fixed/still open/changed.
- Explicitly state retained real-quality gate status.
