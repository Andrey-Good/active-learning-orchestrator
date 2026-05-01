# R24 - Review Tie/Jitter Diagnostics

## Relation to Overall Task
Independent review of W16 tie/near-tie diagnostics and deterministic jitter experiment.

## Assumptions and Resolved Ambiguities
- W16 should not have edited SDK source.
- Jitter variants are benchmark-only.
- Claimed result: near-ties are not meaningful; jitter line rejected.

## Goal
Validate implementation, artifacts, near-tie calculations, overlap/quality gates, and rejection decision.

## Responsibility Boundaries
- Read-only review.
- Do not edit files.

## In Scope
- Review `benchmarks/run_learning_curve_experiments.py` tie/jitter code.
- Review `benchmarks/results/learning_curves/tie_jitter_*`.
- Recompute:
  - row counts and unique keys;
  - score diagnostics;
  - near-tie rates;
  - parent overlap;
  - quality/skew gates.
- Verify no SDK source changes from W16.

## Out of Scope
- Do not implement fixes.
- Do not run varying confirmation.
- Do not suggest new methods as blocking.

## Files/Areas May Read
- `benchmarks/run_learning_curve_experiments.py`
- `benchmarks/results/learning_curves/tie_jitter_*`
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
- Depends on W16.

## Parallel/Sequential Notes
- Must complete before final optimization-cycle report.
