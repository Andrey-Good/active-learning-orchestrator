# R22 - Review Mix Scheduler Exclusion

## Relation to Overall Task
Independent review of W15 exclusion-aware mix scheduler changes.

## Assumptions and Resolved Ambiguities
- W15 was allowed to edit only `src/active_learning_sdk/engine.py` and focused tests.
- Goal is correctness of existing `mode="mix"`, not benchmark artifact changes.

## Goal
Validate mix scheduler behavior, snapshot contents, and tests.

## Responsibility Boundaries
- Read-only review.
- Do not edit files.

## In Scope
- Review `StrategyScheduler.select_batch` mix branch.
- Review tests added/modified in `tests/test_core_sdk.py`.
- Check overlap/underfill/fallback edge cases.
- Run tests if feasible.

## Out of Scope
- Do not review unrelated engine changes.
- Do not benchmark quality.
- Do not implement fixes.

## Files/Areas May Read
- `src/active_learning_sdk/engine.py`
- `tests/test_core_sdk.py`

## Files/Areas Must Not Touch
- Entire repo is read-only.

## Acceptance Criteria
- If clean, explicitly state no remaining in-scope defects, blocking risks, blocking questions, or required improvement requests.
- If not clean, provide exact blocking findings and expected fixes.

## Expected Tests and Validations
- `uv run --group dev pytest tests/test_core_sdk.py tests/test_strategy_correctness.py -q`
- Optional full pytest.

## Dependencies
- Depends on W15.

## Parallel/Sequential Notes
- Can run while W14 continues.
