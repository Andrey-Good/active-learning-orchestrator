# Task ID: 2026-04-24-r07-review-engine-fix-round

## Goal

Re-review the engine after the pool-exhaustion fix.

## Scope

Read-only review of:

- `src/active_learning_sdk/engine.py`
- `tests/test_core_sdk.py`

## Specific Questions

- Does pool exhaustion still create a zero-selection `DONE` round?
- Does `max_rounds` still count only real completed learning rounds?
- Does budget clipping still work?
- Do the tests prove the edge cases?

## Acceptance Criteria

Return findings first. If no blocking findings remain, state that explicitly.
