# R21 - Review Predicted-Class-Balanced Sweep

## Relation to Overall Task
Independent review of W13 benchmark-only predicted-class-balanced uncertainty experiment.

## Assumptions and Resolved Ambiguities
- W13 should not have changed SDK source.
- `pcb_*` variants are benchmark-only and should not be promoted unless metrics pass gates.
- Claimed result: hypothesis rejected; no `pcb_*` beats random and no variant passes fixed gate.

## Goal
Validate implementation, artifacts, metric math, and rejection/promotion decision.

## Responsibility Boundaries
- Read-only review.
- Do not edit files.

## In Scope
- Review `benchmarks/run_learning_curve_experiments.py` PCB code.
- Review `benchmarks/results/learning_curves/pcb_fixed_*`.
- Recompute row counts, unique keys, statuses, AULC/final metrics, lift math, skew deltas, and decision gate.
- Verify no SDK source changes were made by W13.

## Out of Scope
- Do not implement fixes.
- Do not run varying confirmation unless needed to validate a claimed pass.
- Do not suggest new methods as blocking.

## Files/Areas May Read
- `benchmarks/run_learning_curve_experiments.py`
- `benchmarks/results/learning_curves/pcb_fixed_*`
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
- Depends on W13.

## Parallel/Sequential Notes
- Must complete before PCB rejection is accepted.
