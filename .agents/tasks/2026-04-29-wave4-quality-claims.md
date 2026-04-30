# Task: wave4-quality-claims

## Context
Repeat black-box benchmark stress after another supposed fix round. Prior Wave3 still failed tiny/capped Banking77 and Emotion quality gates. README now cites retained passed capped gates with specific budgets.

## Goal
Retest real-data quality claims and search for new benchmark/metric failures.

## Responsibility Boundaries
Owns only `.agents/tmp/blackbox_stress/wave4_quality/` and `.agents/tmp/blackbox_stress/wave4_quality_findings.md`.

## In Scope
- Read public docs and prior reports.
- Use documented benchmark CLI only; do not inspect benchmark implementation source.
- Re-run a bounded Banking77 matrix close to README retained claim if feasible: budgets including `100,300,500`, seeds as documented or stronger, explicit train/test caps.
- Re-run Emotion matrix close to README retained claim if feasible: budgets including `50,100,200`, seeds `13,21,34`, caps.
- Run at least one adversarial tiny-budget matrix for budget warnings and aliasing.
- Analyze quality gates, final lift, AULC lift, non-loss rates, fallback/aliasing diagnostics, calibration, runtime, and README claim consistency.

## Out of Scope
- Reading SDK source or benchmark implementation source.
- Modifying benchmark source/results outside owned output dirs.
- Making broad uncapped claims from capped runs.

## Must Not Touch
- `src/**`
- benchmark source files
- promoted result dirs
- sibling wave4 directories

## Acceptance Criteria
- At least 2 real benchmark commands attempted, unless dependency/network failure blocks them.
- At least 4 strategy families covered.
- Explicitly state whether Wave3 real-quality failures are fixed/still open/changed.
- Explicitly state whether README retained claim settings reproduce.
