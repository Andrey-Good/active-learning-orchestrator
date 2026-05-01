# W55 - Fix Sklearn Adapter Probability Validation

## Context
R69 found two P2 bugs in `SklearnTextClassifierAdapter`:
- `predict_proba()` silently accepts fewer output rows than input texts.
- `_validate_probability_rows()` accepts negative probability values.

## Goal
Make sklearn adapter probability validation strict and aligned with the SDK strategy probability contract.

## Responsibility Boundaries
Own only sklearn adapter validation and tests.

## In Scope
- `src/active_learning_sdk/adapters/sklearn.py`
- `tests/test_sklearn_adapter.py`

## Out of Scope
- Do not edit capability contract files.
- Do not edit engine, strategies, README, benchmarks, or dependency files.

## Required Fixes
- `predict_proba(texts)` must return exactly one probability row per input text.
- If estimator output row count differs from input count, raise `ModelAdapterError` with a clear message.
- `_validate_probability_rows()` must reject negative values before normalization.
- `evaluate()` should convert adapter probability errors into `ModelAdapterError`, not leak raw sklearn errors when probability row count is invalid.
- Keep valid `decision_function` fallback behavior intact.

## Tests
Add tests for:
- Short `predict_proba` output raises `ModelAdapterError`.
- Negative probability values raise `ModelAdapterError`.
- Existing tests still pass.

## Validation
- `uv run --group dev pytest -q tests/test_sklearn_adapter.py`
- `uv run --group dev pytest -q`

## Forbidden Actions
- Do not run destructive git commands.
- Do not revert unrelated changes.

## Acceptance Criteria
- R69 findings are fixed.
- Full tests pass.
