# Task ID: 2026-04-24-w01-fix-completed-round-budget-semantics

## Relation To Overall Task

This is the first implementation task in the scientific optimization program. Current SDK-vs-local benchmark metrics are confounded because `max_rounds` stops based on created rounds, not completed train/eval/update cycles.

## Goal

Fix SDK run semantics so `StopCriteria(max_rounds=N)` means N completed rounds, not N created/selected rounds, and so `budget` / `max_labeled` clips the next selected batch to the remaining label budget where possible.

## Responsibility Boundaries

Own SDK engine behavior and focused tests for this issue.

## In Scope

- `src/active_learning_sdk/engine.py`
- `tests/test_core_sdk.py`
- adjacent tests only if needed

## Out Of Scope

- benchmark notebooks;
- benchmark CSV artifacts;
- README;
- new active learning methods;
- large refactors.

## Files That May Be Changed

- `src/active_learning_sdk/engine.py`
- `tests/test_core_sdk.py`

## Files That Must Not Be Touched

- `benchmarks/**`
- `active_learning_lab.ipynb`
- `lab/**`
- `README.md`
- strategy implementations unless strictly required for this bug

## Architectural Constraints

- Preserve `run_step()` as a single-step primitive.
- Keep `run()` as the blocking loop.
- Do not silently create partial trailing rounds when `max_rounds` is reached.
- Existing unfinished rounds should still be resumable and finishable.
- `max_labeled` should prevent selecting more than the remaining budget when the next step is a new SELECT.

## Execution Plan

1. Reproduce current issue with a focused test: `project.run(stop_criteria=StopCriteria(max_rounds=3), batch_size=1 or 2)` should complete exactly 3 rounds.
2. Add a focused budget clipping test: `budget=3`, `batch_size=2` should not label 4 samples.
3. Patch engine stop logic to count completed rounds.
4. Patch `run()` to pass an effective batch size based on remaining `max_labeled` budget.
5. Run focused tests.
6. Run full test suite if focused tests pass.

## Acceptance Criteria

- `max_rounds=3` yields 3 `DONE` rounds in the oracle/no-wait test path.
- No extra selected-only tail round is created after reaching max completed rounds.
- `budget` / `max_labeled` does not overshoot by selecting a full batch when fewer labels remain.
- Existing tests still pass.

## Expected Tests

- New test for completed-round semantics.
- New test for budget clipping.
- Existing `uv run --group dev pytest -q` or at least `uv run --group dev pytest tests/test_core_sdk.py -q` if full suite is slow.
