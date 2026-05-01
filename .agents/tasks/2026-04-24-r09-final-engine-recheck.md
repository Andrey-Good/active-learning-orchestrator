# Task ID: 2026-04-24-r09-final-engine-recheck

## Goal

Final read-only review of engine completed-round and budget semantics after the second fix round.

## Scope

- `src/active_learning_sdk/engine.py`
- `tests/test_core_sdk.py`

## Check

- `_completed_round_count()` ignores empty `DONE` rounds.
- `max_rounds` means completed learning rounds.
- pool exhaustion does not create zero-selection completed rounds.
- budget clipping is tested with `budget < pool_size`.

## Acceptance Criteria

Return findings first. If no blocking findings remain, state that explicitly.
