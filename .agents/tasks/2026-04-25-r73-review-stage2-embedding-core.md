# R73 - Review Stage 2 Embedding Core

## Context
W57 implemented Stage 2 embedding strategies and cache key scoping.

## Goal
Review the embedding core for correctness, deterministic behavior, cache safety, and integration with Stage 1 capability validation.

## Responsibility Boundaries
- This is a read-only review.
- Focus on W57-owned files and Stage 2 core acceptance criteria.

## In Scope
- `src/active_learning_sdk/cache.py`
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/strategies/__init__.py`
- `src/active_learning_sdk/strategies/uncertainty.py`
- `src/active_learning_sdk/strategies/embedding.py`
- `tests/test_embedding_strategies.py`
- `tests/test_strategy_capabilities.py`

## Out of Scope
- Do not edit files.
- Do not review benchmark diagnostics; they are a later Stage 2 subtask.
- Do not implement BADGE/stochastic/committee methods.

## Review Questions
- Is `coreset_kcenter` now a real strategy, not a placeholder?
- Are all five embedding strategies registered and exported?
- Do they declare `required_capabilities={"embed"}`?
- Does configure/attach accept embedding-capable models and fail without `embed`?
- Are malformed embeddings rejected with `ConfigurationError`?
- Are selections deterministic and duplicate-free?
- Does k-center use labeled ids as existing centers when present?
- Are cache keys scoped by model id, dataset fingerprint, embedding config/version, and sample id without breaking backward compatibility?
- Are tests strong enough for edge cases listed in W57?

## Validation
- `uv run --group dev pytest -q tests/test_embedding_strategies.py tests/test_strategy_capabilities.py`
- `uv run --group dev pytest -q`

## Forbidden Actions
- Do not edit files.
- Do not run destructive git commands.

## Acceptance Criteria
- No blocking findings remain for Stage 2 core.
- Full tests pass.
