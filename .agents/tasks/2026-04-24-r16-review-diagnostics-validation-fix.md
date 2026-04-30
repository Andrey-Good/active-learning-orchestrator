# R16 - Review Diagnostics Validation Fix

## Relation to Overall Task
Independent re-review of W10, which fixed R15 blocking findings in acquisition diagnostics validation.

## Assumptions and Resolved Ambiguities
- R15 required fingerprint validation before rebuilding label diagnostics.
- R15 required overlap-key drift to fail loudly instead of silently producing zero overlaps.
- W10 claims both are fixed.

## Goal
Confirm diagnostics validation is now robust enough to support subsequent optimization experiments.

## Responsibility Boundaries
- Read-only review.
- Do not edit files.

## In Scope
- Inspect `benchmarks/run_learning_curve_experiments.py` diagnostics label-map and overlap code.
- Inspect regenerated `acquisition_*` artifacts.
- Run/recompute targeted validation if feasible.
- Verify R15 findings are resolved.

## Out of Scope
- No new diagnostics.
- No strategy changes.
- No fixes.

## Files/Areas May Read
- `benchmarks/run_learning_curve_experiments.py`
- `benchmarks/results/learning_curves/acquisition_*`
- `benchmarks/results/learning_curves/baseline_sweep_runs.csv`

## Files/Areas Must Not Touch
- Entire repo is read-only.

## Execution Plan
- Inspect fingerprint validation code -> verify raw fingerprint uniqueness/match is enforced.
- Inspect overlap code -> verify missing keys cannot silently become empty sets.
- Run diagnostics CLI or artifact script -> verify current artifacts pass.

## Acceptance Criteria
- If clean, explicitly state no remaining in-scope defects, blocking risks, blocking questions, or required improvement requests.
- If not clean, provide exact blocking findings and expected fixes.

## Expected Tests and Validations
- `uv run python benchmarks/run_learning_curve_experiments.py --diagnostics --diagnostics-runs-path benchmarks/results/learning_curves/baseline_sweep_runs.csv` if feasible.
- Artifact recompute/integrity check.

## Dependencies
- Depends on W10.

## Parallel/Sequential Notes
- Must complete before running optimization experiments based on diagnostics.
