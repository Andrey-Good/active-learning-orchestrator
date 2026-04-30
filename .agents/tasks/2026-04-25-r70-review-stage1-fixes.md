# R70 - Review Stage 1 Fix Loop

## Context
W54 and W55 fixed review findings from R68/R69:
- Protocol-inherited adapter stubs were incorrectly treated as real capabilities.
- Sklearn adapter accepted malformed probability output.

## Goal
Verify the fixes close the reported findings without regressions.

## Responsibility Boundaries
- This is a read-only review.
- Focus on exact R68/R69 repro cases and Stage 1 integration.

## In Scope
- `src/active_learning_sdk/adapters/base.py`
- `src/active_learning_sdk/adapters/sklearn.py`
- `tests/test_strategy_capabilities.py`
- `tests/test_sklearn_adapter.py`

## Out of Scope
- Do not edit files.
- Do not implement new features.
- Do not review unrelated dirty worktree changes.

## Required Checks
- Reproduce/verify that a class subclassing `TextClassificationAdapter` without overriding `predict_proba` is reported as missing `predict_proba`.
- Verify entropy scheduler configure fails with the new reason.
- Verify sklearn adapter rejects short probability output with `ModelAdapterError`.
- Verify sklearn adapter rejects negative probabilities with `ModelAdapterError`.
- Verify full tests pass.

## Validation
- `uv run --group dev pytest -q tests/test_strategy_capabilities.py tests/test_sklearn_adapter.py`
- `uv run --group dev pytest -q`

## Forbidden Actions
- Do not edit files.
- Do not run destructive git commands.

## Acceptance Criteria
- No findings remain for R68/R69 issues.
- Stage 1 can proceed to integration/public smoke validation.
