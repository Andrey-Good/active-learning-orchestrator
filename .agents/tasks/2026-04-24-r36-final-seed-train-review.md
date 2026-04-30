# R36 - Final Seed Train Review

## Relation To Overall Task
W26 fixed R35's remaining runtime rebind issue for seed train before first select. This review gates accepting the SDK seed-train improvement.

## Assumptions And Resolved Ambiguities
- R34 restart defect and R35 same-engine runtime rebind defect should now be fixed.
- Seed train may be repeated after runtime rebind/restart because model state is not persisted.
- Same-runtime no-repeat without rebind should still hold.

## Goal And Expected Result
Perform read-only final review of seed train before first select. Explicitly state whether any in-scope blocking defects, risks, questions, or required improvements remain.

## Responsibility Boundaries
Read-only. Do not edit files.

## In Scope
- `src/active_learning_sdk/engine.py`
- `tests/test_import_labels.py`
- Restart and attach_runtime rebind behavior.
- Same-runtime no-repeat behavior.
- No fake rounds/tasks.

## Out Of Scope
- Benchmark harness update to remove warm-start.
- Model persistence.
- New strategies.

## Files Or Modules May Be Changed
None.

## Files Or Areas Must Not Be Touched
Entire repository.

## Important Architectural Constraints And Forbidden Actions
- Keep optional future model persistence separate.
- Do not require more than available tests unless a concrete defect is found.

## High-Level Execution Plan
- Inspect W26 changes.
- Optionally run targeted tests.
- Report final status.

## Acceptance Criteria
- R34 and R35 issues are fixed.
- No new blocking issue remains.

## Expected Tests And Validations
Optional:
- `uv run --group dev pytest tests/test_import_labels.py -q`

## Dependencies
Depends on W26.

## Parallel Or Sequential Notes
Sequential before benchmark project smoke update.
