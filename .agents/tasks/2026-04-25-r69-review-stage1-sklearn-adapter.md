# R69 - Review Stage 1 Sklearn Adapter

## Context
Worker W53 implemented `SklearnTextClassifierAdapter` as the first real fast training adapter.

## Goal
Review adapter behavior, tests, exports, and integration with Stage 1 capability contracts.

## Responsibility Boundaries
- This is a read-only review.
- Focus on W53-owned files and public adapter behavior.

## In Scope
- `src/active_learning_sdk/adapters/sklearn.py`
- `src/active_learning_sdk/adapters/__init__.py`
- `tests/test_sklearn_adapter.py`
- Integration with current adapter capability inspection.

## Out of Scope
- Do not edit files.
- Do not review or refactor engine capability validation except where it breaks sklearn adapter use.
- Do not update README or benchmarks.

## Review Questions
- Does the default adapter train quickly and deterministically?
- Does it expose `fit`, `predict_proba`, `evaluate`, and `get_model_id` correctly?
- Does `get_model_id()` change after successful fit?
- Are probability outputs valid, finite, row-shaped, and summing to 1?
- Does fallback for estimators without `predict_proba` behave correctly or fail clearly?
- Are errors raised as `ModelAdapterError` where appropriate?
- Is the adapter exported from `active_learning_sdk.adapters`?
- Do tests cover injected estimators and edge cases?

## Validation
- Run `uv run --group dev pytest -q tests/test_sklearn_adapter.py`.
- Run full `uv run --group dev pytest -q` if feasible.

## Forbidden Actions
- Do not edit files.
- Do not run destructive git commands.

## Acceptance Criteria
- No blocking sklearn adapter findings.
- Validation passes in the combined current workspace.
