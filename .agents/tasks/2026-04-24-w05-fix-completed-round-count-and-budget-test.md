# Task ID: 2026-04-24-w05-fix-completed-round-count-and-budget-test

## Goal

Address engine re-review findings after the pool-exhaustion fix.

## Scope

You own only:

- `src/active_learning_sdk/engine.py`
- `tests/test_core_sdk.py`

## Findings To Fix

1. `_completed_round_count()` counts all `DONE` rounds, including historical or corrupted zero-selection `DONE` rounds. It should count only real completed learning rounds, likely `status == DONE` and non-empty `selected_sample_ids`.
2. The budget clipping test currently uses a budget larger than pool size, so it proves pool exhaustion rather than remaining-budget clipping. Strengthen it with pool size greater than budget, e.g. 4 samples, `budget=3`, `batch_size=2`.

## Out Of Scope

- benchmark files;
- notebooks;
- strategy changes;
- README.

## Execution Plan

1. Update completed-round counting to ignore empty selected batches.
2. Strengthen tests for true budget clipping below pool size.
3. Add a regression test or assertion for historical empty `DONE` round if practical.
4. Run focused and full tests.

## Acceptance Criteria

- Empty `DONE` rounds do not count toward `max_rounds`.
- Budget clipping test proves `budget < pool_size` and no overshoot.
- `uv run --group dev pytest tests/test_core_sdk.py -q` passes.
- `uv run --group dev pytest -q` passes if feasible.
