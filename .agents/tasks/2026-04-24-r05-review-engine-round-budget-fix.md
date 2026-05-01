# Task ID: 2026-04-24-r05-review-engine-round-budget-fix

## Relation To Overall Task

Independent review of the first engine fix in the scientific optimization program.

## Goal

Verify that the engine fix correctly changes `max_rounds` to completed-round semantics and clips label budget without introducing regressions.

## Responsibility Boundaries

Read-only review. Do not edit files.

## In Scope

- `src/active_learning_sdk/engine.py`
- `tests/test_core_sdk.py`
- behavior around `run()`, `run_step()`, `_should_stop()`, and `StopCriteria`

## Out Of Scope

- benchmark runner;
- notebooks;
- new AL methods.

## Files That May Be Changed

None.

## Files That Must Not Be Touched

The entire repository. Review only.

## Review Checklist

- Does `max_rounds=N` now mean N completed `DONE` rounds?
- Does `run()` avoid creating selected-only tail rounds after max completed rounds?
- Does budget clipping happen only where safe?
- Does the behavior preserve unfinished-round resume?
- Are tests sufficient?
- Are there edge cases for `batch_size <= 0`, `budget=0`, pool exhaustion, or active unfinished rounds?

## Acceptance Criteria

- Return findings first, ordered by severity.
- If no blocking findings remain, explicitly state that.
