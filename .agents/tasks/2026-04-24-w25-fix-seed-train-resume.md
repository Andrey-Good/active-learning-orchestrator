# W25 - Fix Seed Train Resume Safety

## Relation To Overall Task
R34 found a blocking defect in W24: after seed TRAIN_EVAL is persisted, the SDK does not persist model weights. If the process restarts before first SELECT, a fresh runtime model skips seed training because `metrics_history` exists and uncertainty selection fails on an unfitted model.

## Assumptions And Resolved Ambiguities
- R34 finding is valid.
- The SDK currently does not persist model weights; do not invent model persistence in this task.
- Repeating seed train/eval before first SELECT is acceptable because it has no backend side effects and no fake rounds/tasks.
- Avoid infinite repeated seed training in the same process if the model is already trained and next call can select.

## Goal And Expected Result
Make seed train/eval resume-safe. A project reopened after seed TRAIN_EVAL but before first SELECT should train the fresh runtime model on imported seed labels before selecting, rather than failing during uncertainty prediction.

## Responsibility Boundaries
Owned by this worker:
- `src/active_learning_sdk/engine.py`
- `tests/test_import_labels.py` or a focused test file under `tests/**`

Do not change:
- `benchmarks/**`
- strategy implementations
- docs/root README

## In Scope
- Adjust seed-train guard so persisted `seed_eval` alone does not suppress training for a fresh runtime model before first active selection.
- Track per-process seed training state if needed.
- Add restart test: import labels, run seed train, close/reopen/attach fresh model, next `run_step` trains again or otherwise ensures model is fitted before SELECT; following `run_step` selects successfully.
- Preserve no-repeat behavior in a single process.

## Out Of Scope
- Model persistence/save-load integration.
- Benchmark smoke update.
- New public API.

## Files Or Modules May Be Changed
- `src/active_learning_sdk/engine.py`
- `tests/**`

## Files Or Areas Must Not Be Touched
- `benchmarks/**`
- `README.md`
- `docs/**`
- `docker/**`
- `.git/**`

## Important Architectural Constraints And Forbidden Actions
- Do not create fake rounds.
- Do not silently catch model prediction failures as a fallback to random.
- Do not mutate private benchmark code.
- Do not break existing `run()` stop criteria.

## High-Level Execution Plan
- Add an engine instance flag recording that seed training has run in this process/runtime.
- Change `_should_seed_train_before_select` to allow seed train when no active rounds have selected samples and this runtime has not done seed training, even if prior seed_eval metrics exist.
- Set the flag after successful seed train/eval.
- Add tests for restart and single-process no-repeat.

## Step -> Verify Plan
- Implement runtime flag -> verify existing no-repeat test still passes.
- Add restart test -> verify reopened fresh model gets seed-trained before select.
- Run full available tests.

## Acceptance Criteria
- R34 defect is fixed.
- Tests cover the restart case.
- No benchmark files changed.

## Expected Tests And Validations
- `uv run --group dev pytest tests/test_import_labels.py -q`
- `uv run --group dev pytest -q`

## Dependencies
Depends on W24/R34.

## Parallel Or Sequential Notes
Sequential fix before reviewing seed-train feature again.
