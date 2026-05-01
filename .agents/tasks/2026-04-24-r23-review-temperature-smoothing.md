# R23 - Review Temperature Smoothing Sweep

## Relation to Overall Task
Independent review of W14 benchmark-only temperature smoothing experiment.

## Assumptions and Resolved Ambiguities
- W14 should not have changed SDK source.
- Temperature variants are benchmark-only.
- Claimed result: all variants informative; `least_confidence_T2` passes parent quality gate but not random/promotion gate.

## Goal
Validate implementation, artifacts, informative/quality/promotion decisions, and summary math.

## Responsibility Boundaries
- Read-only review.
- Do not edit files.

## In Scope
- Review temperature strategy implementation in `benchmarks/run_learning_curve_experiments.py`.
- Review `benchmarks/results/learning_curves/temperature_*`.
- Recompute:
  - row counts and unique keys;
  - AULC/final metrics;
  - overlap vs parent;
  - quality gates;
  - promotion gate vs random.
- Verify no SDK source changes from W14.

## Out of Scope
- Do not implement fixes.
- Do not run varying confirmation.
- Do not suggest new methods as blocking.

## Files/Areas May Read
- `benchmarks/run_learning_curve_experiments.py`
- `benchmarks/results/learning_curves/temperature_*`
- `git status --short`

## Files/Areas Must Not Touch
- Entire repo is read-only.

## Acceptance Criteria
- If clean, explicitly state no remaining in-scope defects, blocking risks, blocking questions, or required improvement requests.
- If not clean, list exact blocking findings and expected fixes.

## Expected Tests and Validations
- Artifact recompute script.
- Optional py_compile/help smoke.

## Dependencies
- Depends on W14.

## Parallel/Sequential Notes
- Must complete before deciding next quality experiment.
