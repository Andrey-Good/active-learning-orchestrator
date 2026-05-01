# R76 - Review Stage 3 BADGE Core

## Context
W60 implemented `badge` as a built-in strategy requiring `gradient_embed`.

## Goal
Review BADGE core implementation for capability correctness, deterministic geometry, validation, and integration with scheduler/configuration.

## Responsibility Boundaries
- This is a read-only review.
- Focus on W60-owned BADGE core files.

## In Scope
- `src/active_learning_sdk/strategies/badge.py`
- `src/active_learning_sdk/strategies/__init__.py`
- `src/active_learning_sdk/engine.py`
- `tests/test_badge_strategy.py`
- `tests/test_strategy_capabilities.py`

## Out of Scope
- Do not edit files.
- Do not review benchmark wiring; it is a later Stage 3 subtask.
- Do not implement neural adapters.

## Review Questions
- Does `badge` require `gradient_embed` and fail fast when missing?
- Does `SelectionContext.gradient_embed` mirror error handling expectations from `embed()`/`predict_proba()`?
- Does BADGE validate row count, raggedness, empty rows, non-numeric/bool, and non-finite values?
- Is deterministic k-means++ selection implemented as specified?
- Are duplicate pool ids and k edge cases handled?
- Is `badge` registered in built-ins, lookup, and exports?
- Are tests sufficient?

## Validation
- `uv run --group dev pytest -q tests/test_badge_strategy.py tests/test_strategy_capabilities.py`
- `uv run --group dev pytest -q`

## Forbidden Actions
- Do not edit files.
- Do not run destructive git commands.

## Acceptance Criteria
- No blocking BADGE core findings remain.
