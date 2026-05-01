# Task ID: 2026-04-24-w03-fix-pool-exhaustion-fake-round

## Relation To Overall Task

Fix round after independent review of the completed-round/budget engine change.

## Goal

Prevent pool exhaustion from creating and counting a zero-selection fake `DONE` round.

## Responsibility Boundaries

Own only the engine/test fix for this review finding.

## In Scope

- `src/active_learning_sdk/engine.py`
- `tests/test_core_sdk.py`

## Out Of Scope

- benchmark runner;
- notebooks;
- unrelated strategy changes.

## Review Finding To Address

When the unlabeled pool is exhausted, `run_step()` can append a new round before `_step_select()` discovers there are no samples. `_step_select()` then marks that empty round `DONE`, which makes completed-round accounting count a non-learning round.

## Files That May Be Changed

- `src/active_learning_sdk/engine.py`
- `tests/test_core_sdk.py`

## Files That Must Not Be Touched

- `benchmarks/**`
- notebooks
- README

## Execution Plan

1. Add/reproduce a test where dataset has 3 samples, `run(max_rounds=4, batch_size=1)` completes only 3 real rounds and does not leave a zero-selection `DONE` round.
2. Add/reproduce a test where `budget=4`, `batch_size=2`, 3-sample pool does not leave selected lengths `[2, 1, 0]`.
3. Patch round creation or empty-pool handling so no fake completed round remains.
4. Run focused and full tests.

## Acceptance Criteria

- No `DONE` round with empty `selected_sample_ids` is created for pool exhaustion.
- Completed-round count reflects real selected/trained rounds.
- Tests prove the edge case.

## Expected Validation

- `uv run --group dev pytest tests/test_core_sdk.py -q`
- `uv run --group dev pytest -q`
