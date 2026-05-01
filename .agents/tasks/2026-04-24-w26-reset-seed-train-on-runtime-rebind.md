# W26 - Reset Seed Train Flag On Runtime Rebind

## Relation To Overall Task
R35 found one remaining seed-train edge case: `attach_runtime()` can bind a fresh model into the same engine after seed training and before first SELECT, but `_seed_train_completed_in_runtime` remains true and suppresses necessary retraining.

## Assumptions And Resolved Ambiguities
- R35 finding is valid.
- The seed train flag should describe the currently bound runtime model, not just the engine object.
- Resetting it on `attach_runtime()` is acceptable because attach_runtime explicitly rebinds runtime objects.
- Avoid unnecessary reset on unrelated state reads.

## Goal And Expected Result
Reset or re-key seed-train completion when runtime/model is rebound so fresh models are trained on imported seed labels before uncertainty selection.

## Responsibility Boundaries
Owned by this worker:
- `src/active_learning_sdk/engine.py`
- `tests/test_import_labels.py` or focused tests under `tests/**`

Do not change:
- `benchmarks/**`
- strategies
- docs/root README

## In Scope
- Reset `_seed_train_completed_in_runtime` in `attach_runtime()` when model is rebound.
- Consider reset in `configure()` if needed for symmetry.
- Add test covering same-engine `attach_runtime()` rebinding fresh model after seed train and before first SELECT.

## Out Of Scope
- Model persistence.
- Benchmark update.
- Public API changes.

## Files Or Modules May Be Changed
- `src/active_learning_sdk/engine.py`
- `tests/**`

## Files Or Areas Must Not Be Touched
- `benchmarks/**`
- `README.md`
- `docs/**`
- `docker/**`

## Important Architectural Constraints And Forbidden Actions
- Do not disable same-runtime no-repeat behavior without runtime rebind.
- Do not create fake rounds.
- Do not catch prediction failures as fallback.

## High-Level Execution Plan
- Update runtime binding path to reset seed-train flag.
- Add regression test for same-engine rebind.
- Run targeted and full tests.

## Acceptance Criteria
- R35 required improvement fixed.
- Existing restart and no-repeat tests still pass.
- Full available test suite passes.

## Expected Tests And Validations
- `uv run --group dev pytest tests/test_import_labels.py -q`
- `uv run --group dev pytest -q`

## Dependencies
Depends on W25/R35.

## Parallel Or Sequential Notes
Sequential before final seed-train review.
