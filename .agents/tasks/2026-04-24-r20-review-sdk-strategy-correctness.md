# R20 - Review SDK Strategy Correctness

## Relation to Overall Task
Independent review of W12 SDK correctness changes for built-in strategies.

## Assumptions and Resolved Ambiguities
- W12 was allowed to change only `src/active_learning_sdk/strategies/uncertainty.py` and focused tests.
- W12 should not have changed benchmark runner, engine, README, or Docker.
- Goal is safe deterministic behavior, not benchmark metric lift.

## Goal
Find any defects, regressions, or missing tests in probability validation, normalization, deterministic tie-breaking, and deterministic random selection.

## Responsibility Boundaries
- Read-only review.
- Do not edit files.

## In Scope
- Review `src/active_learning_sdk/strategies/uncertainty.py`.
- Review `tests/test_strategy_correctness.py`.
- Run focused/full tests if feasible.
- Check behavior for edge cases:
  - row count mismatch;
  - empty probability row;
  - NaN/inf/negative/zero-sum;
  - non-normalized rows;
  - equal score tie;
  - global RNG independence.

## Out of Scope
- Do not review benchmark quality.
- Do not suggest new methods as blocking.
- Do not edit files.

## Files/Areas May Read
- `src/active_learning_sdk/strategies/uncertainty.py`
- `tests/test_strategy_correctness.py`
- `tests/test_core_sdk.py` if needed.

## Files/Areas Must Not Touch
- Entire repo is read-only.

## Acceptance Criteria
- If clean, explicitly state no remaining in-scope defects, blocking risks, blocking questions, or required improvement requests.
- If not clean, give exact blocking findings and expected fixes.

## Expected Tests and Validations
- `uv run --group dev pytest tests/test_strategy_correctness.py -q`
- `uv run --group dev pytest -q`

## Dependencies
- Depends on W12.

## Parallel/Sequential Notes
- Must pass before SDK correctness changes are considered accepted.
