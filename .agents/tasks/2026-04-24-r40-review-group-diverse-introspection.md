# R40 - Review Group-Diverse Introspection Fix

## Relation To Overall Task
W30 fixed the non-blocking R39 issue by registering `GroupDiverseEntropyStrategy()` in scheduler built-ins for introspection.

## Goal
Read-only review that the fix is correct and did not regress selection behavior.

## Scope
Inspect:
- `src/active_learning_sdk/engine.py`
- `tests/test_group_diverse_strategy.py`

Do not edit files.

## Acceptance Criteria
- Shared built-in factory includes group-diverse entropy.
- Configure and attach runtime paths use it.
- Tests are adequate.
- No blocking issues remain.

## Expected Validation
Optional:
- `uv run --group dev pytest tests/test_group_diverse_strategy.py -q`
