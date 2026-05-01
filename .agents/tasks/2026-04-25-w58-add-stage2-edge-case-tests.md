# W58 - Add Explicit Stage 2 Embedding Edge-Case Tests

## Context
R73 found a non-blocking but valid coverage gap: empty pool, `k <= 0`, and partial duplicate embeddings are handled by implementation but not explicitly tested across embedding strategies.

## Goal
Add explicit tests for these Stage 2 edge cases without changing implementation unless a test reveals a real bug.

## Responsibility Boundaries
Own only embedding strategy tests unless a real defect is found.

## In Scope
- `tests/test_embedding_strategies.py`
- `src/active_learning_sdk/strategies/embedding.py` only if a newly added test exposes a bug that must be fixed.

## Out of Scope
- Do not edit benchmarks.
- Do not edit engine/cache/docs/dependencies.

## Required Tests
- Empty pool returns `[]` for every embedding strategy.
- `k <= 0` returns `[]` for every embedding strategy.
- Partial duplicate embeddings return deterministic unique ids and do not select duplicate sample ids.
- If useful, assert duplicate-heavy selection behavior for `deduplicate_near_neighbors` specifically.

## Validation
- `uv run --group dev pytest -q tests/test_embedding_strategies.py`
- `uv run --group dev pytest -q`

## Forbidden Actions
- Do not run destructive git commands.
- Do not revert unrelated changes.

## Acceptance Criteria
- R73 P3 coverage gap is closed.
- Full tests pass.
