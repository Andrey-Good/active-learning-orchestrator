# R34 - Review Seed Train Before First Select

## Relation To Overall Task
W24 changed the SDK engine so imported seed labels can trigger train/eval before the first active SELECT. This should remove the need for external model warm-starting and improve cold-start active-learning correctness.

## Assumptions And Resolved Ambiguities
- Seed train/eval should not create fake backend rounds/tasks.
- Returning `StepKind.TRAIN_EVAL` with `round_id=None` is allowed only if it is coherent and documented enough for callers.
- Seed train/eval should happen once when needed, not repeatedly.
- Existing normal round train/eval behavior must remain correct.

## Goal And Expected Result
Perform read-only review of W24 changes. Check correctness, state-machine safety, stop criteria interaction, metric history semantics, tests, and compatibility risks.

## Responsibility Boundaries
Read-only. Do not edit files.

## In Scope
- `src/active_learning_sdk/engine.py` seed train implementation.
- `tests/test_import_labels.py` new tests.
- Behavior after `import_labels` then `run_step`/`run`.
- No-label path remains unchanged.
- No fake rounds/tasks.
- Whether normal round training still writes round metrics/reward correctly.

## Out Of Scope
- Benchmark harness update to remove warm-start workaround.
- New strategies.
- Docs rewrite.

## Files Or Modules May Be Changed
None.

## Files Or Areas Must Not Be Touched
Entire repository.

## Important Architectural Constraints And Forbidden Actions
- Do not require changing `StepKind` unless the current representation creates a concrete defect.
- Separate optional API polish from blockers.
- Verify claimed tests if cheap.

## High-Level Execution Plan
- Inspect engine diff and tests.
- Optionally rerun targeted tests.
- Report findings and final status.

## Acceptance Criteria
- No in-scope blocking defects/risks/questions/required improvements remain, or findings are concrete and actionable.

## Expected Tests And Validations
Optional:
- `uv run --group dev pytest tests/test_import_labels.py -q`

## Dependencies
Depends on W24.

## Parallel Or Sequential Notes
Sequential before removing project smoke warm-start workaround.
