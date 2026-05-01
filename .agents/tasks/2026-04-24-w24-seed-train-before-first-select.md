# W24 - Seed Train Before First Select

## Relation To Overall Task
R33 accepted the full project smoke but identified an SDK product gap: after importing seed labels, the state machine still begins with SELECT before any TRAIN_EVAL step. Uncertainty strategies therefore require external model warm-starting. A real product should train/evaluate on imported seed labels before first active selection when possible.

## Assumptions And Resolved Ambiguities
- `import_labels(...)` is approved and public.
- Imported labels should be enough for the engine to train before first active acquisition.
- This should not create fake backend rounds/tasks.
- The behavior should be explicit and conservative: only run seed train/eval when there are labeled train samples and no completed metrics/round training yet.

## Goal And Expected Result
Add engine support for an initial seed training/evaluation step before the first SELECT when imported labels exist. This removes the need for external benchmark model warm-starting and improves the SDK's cold-start active-learning correctness.

## Responsibility Boundaries
Owned by this worker:
- `src/active_learning_sdk/engine.py`
- `tests/**` focused on seed-train-before-select behavior

Do not change:
- `benchmarks/**` in this task
- `src/active_learning_sdk/strategies/**`
- Docker/backends unrelated to this behavior
- root README/docs

## In Scope
- State-machine behavior before first active round.
- A seed/train/eval step result that is observable enough for users.
- Persist metrics history after seed train/eval.
- Ensure later SELECT uses the trained model.
- Tests showing entropy run from imported labels does not need external model.fit before first selection.

## Out Of Scope
- Benchmark harness update to remove warm-start workaround.
- New strategies.
- Full stop criteria redesign.
- Fake rounds/tasks for seed labels.

## Files Or Modules May Be Changed
- `src/active_learning_sdk/engine.py`
- `tests/**`

## Files Or Areas Must Not Be Touched
- `benchmarks/**`
- `src/active_learning_sdk/strategies/**`
- `README.md`
- `docs/**`
- `docker/**`

## Important Architectural Constraints And Forbidden Actions
- Do not create fake backend rounds for seed labels.
- Do not train repeatedly before every select; seed train should happen only when needed.
- Do not break existing projects with no imported labels.
- Do not hide errors from model.fit/evaluate.
- Preserve stop criteria semantics.

## High-Level Execution Plan
- Inspect current `run_step` and `_step_train_eval` structure.
- Add a private seed-train path that reuses training/evaluation logic without requiring a round selected/pulled state, or refactor shared train/eval helper.
- Add a clear `StepKind` if appropriate, or reuse `TRAIN_EVAL` with `round_id=None` only if this fits public contract.
- Persist a metric record with a seed-training marker.
- Add tests with a model that fails predict_proba until fit; after `import_labels`, `run_step` should train before selection, then the next `run_step` should select successfully.

## Step -> Verify Plan
- Implement seed train trigger -> test first `run_step` after import returns train/eval-type result and increments model fit count.
- Verify next `run_step` selects with uncertainty and no external warm-start.
- Verify no seed train when no labels exist.
- Verify no repeated seed train once metrics exist.

## Acceptance Criteria
- Imported seed labels can train the model before first active acquisition through public `run_step`/`run`.
- Tests cover cold-start uncertainty selection without external model warm-start.
- No benchmark files changed.

## Expected Tests And Validations
- `uv run --group dev pytest tests/test_import_labels.py -q`
- New targeted test file or added tests for seed train behavior.
- `uv run --group dev pytest -q`

## Dependencies
Depends on W22/R32 and R33.

## Parallel Or Sequential Notes
Sequential before removing warm-start workaround from project smoke.
