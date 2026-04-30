# W60 - Stage 3 BADGE Core Strategy

## Context
Stage 3 adds BADGE as a first-class active-learning strategy. Stage 1 already introduced `gradient_embed` capability detection. Stage 2 added deterministic k-means++/diversity geometry that can inform BADGE implementation, but BADGE must use `gradient_embed`, not plain `embed`.

## Goal
Implement `badge` as a deterministic strategy over gradient embeddings with strict validation and capability fail-fast behavior.

## Responsibility Boundaries
Own the SDK core BADGE strategy slice.

## In Scope
- New file `src/active_learning_sdk/strategies/badge.py`
- `src/active_learning_sdk/strategies/__init__.py`
- `src/active_learning_sdk/engine.py`
- New tests in `tests/test_badge_strategy.py`
- `tests/test_strategy_capabilities.py` if needed for BADGE capability validation

## Out of Scope
- Do not edit benchmarks yet.
- Do not edit sklearn adapter.
- Do not edit docs/README.
- Do not edit dependency files.
- Do not implement a neural autograd adapter.

## Required Behavior
- Add built-in strategy name `badge`.
- BADGE must declare `required_capabilities = frozenset({"gradient_embed"})`.
- `configure()` and `attach_runtime()` must accept `badge` only when the model has a real `gradient_embed`.
- Missing `gradient_embed` must fail fast with `ConfigurationError` naming strategy and capability.
- Strategy must call `context.gradient_embed(...)` if available. If `SelectionContext` lacks that method, add it analogously to `embed()`.
- Validate gradient embedding output:
  - row count equals requested pool ids;
  - 2D/ragged rows rejected;
  - empty rows rejected;
  - non-numeric/bool rejected;
  - non-finite rejected.
- Selection must be deterministic and duplicate-free.
- Use deterministic k-means++ style selection over gradient embeddings:
  - first center should be chosen by largest embedding norm with stable tie-break;
  - later centers maximize squared distance to current centers with stable tie-break;
  - for identical embeddings, deterministic unique ids are returned.
- Handle edge cases:
  - empty pool;
  - `k <= 0`;
  - `k > len(pool)`;
  - duplicate pool ids.
- Include clear comments for BADGE approximation: the SDK expects the adapter to supply gradient embeddings, including pseudo-label logic if needed.

## Test Requirements
Add tests covering:
- BADGE selects endpoints/diverse points on simple gradient-embedding geometry.
- Determinism and duplicate-free behavior with identical embeddings and duplicate pool ids.
- Malformed gradient embeddings raise `ConfigurationError`.
- Configuring `badge` without `gradient_embed` fails at configure.
- Configuring `badge` with a gradient-capable model succeeds.
- `StrategyScheduler` can select with `badge`.

## Validation
- `uv run --group dev pytest -q tests/test_badge_strategy.py tests/test_strategy_capabilities.py`
- `uv run --group dev pytest -q`

## Forbidden Actions
- Do not run destructive git commands.
- Do not modify benchmark result artifacts.
- Do not revert unrelated changes.

## Acceptance Criteria
- `badge` is a real built-in strategy.
- Full tests pass.
- BADGE is ready for benchmark wiring in the next subtask.
