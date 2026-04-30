# Task: wave3-quality-real-benchmarks

## Context
The user requested repeat black-box stress testing after reported fixes. Prior wave found Banking77 strategy collapse and weak real-data quality. SDK and benchmark source inspection remains forbidden.

## Goal
Re-test real/capped quality and strategy differentiation, then search for new metric failures across datasets, seeds, budgets, and strategy families.

## Responsibility Boundaries
Owns only artifacts under `.agents/tmp/blackbox_stress/wave3_quality/` and findings in `.agents/tmp/blackbox_stress/wave3_quality_findings.md`.

## In Scope
- Read public docs and prior reports.
- Use documented benchmark CLIs only; do not inspect implementation source.
- Re-test prior Banking77 and Emotion failures.
- Attempt at least one additional real dataset or a documented synthetic adversarial matrix if real dataset access fails.
- Analyze quality gates, final lift, AULC lift, non-loss rates, selection differentiation, cold-start fallback diagnostics if visible in artifacts, calibration, runtime, and omitted/clamped budgets.

## Out of Scope
- Reading `src/**` or benchmark source.
- Modifying SDK or benchmark source.
- Claiming uncapped broad superiority from capped runs.

## Must Not Touch
- `src/**`
- benchmark source files
- promoted benchmark result directories
- other wave3 worker directories

## Execution Plan
1. Run a bounded Banking77 repeat matrix comparable to wave2.
2. Run a bounded Emotion or CLINC matrix if available.
3. Run a small adversarial synthetic matrix to check budget materialization and duplicate/group concentration.
4. Parse artifacts and produce a pass/fail/regression table plus new findings.

## Acceptance Criteria
- At least 2 benchmark commands attempted.
- At least 4 strategy families covered.
- Explicitly state whether prior Banking77 collapse is fixed, still present, or replaced by a different failure.
- No source inspection.
