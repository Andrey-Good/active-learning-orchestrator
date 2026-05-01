# W66 - Stage 5 Hybrid Strategy Framework Core

## Context
Stage 5 turns one-off strategy mixes into configurable hybrid acquisition. Current `mix`/`mix_interleaved` can combine selectors, but there is no score normalization framework, no uncertainty-diversity weighted hybrid, and no prefilter/rerank modes.

## Goal
Implement a safe, testable hybrid framework without regressing existing strategies or scheduler behavior.

## Responsibility Boundaries
Own the core SDK hybrid configuration and scheduler integration.

## In Scope
- `src/active_learning_sdk/configs.py`
- `src/active_learning_sdk/engine.py`
- New file `src/active_learning_sdk/strategies/hybrid.py`
- `src/active_learning_sdk/strategies/__init__.py`
- New tests in `tests/test_hybrid_strategy_framework.py`
- `tests/test_strategy_capabilities.py` if needed

## Out of Scope
- Do not edit benchmarks yet.
- Do not edit docs/README yet.
- Do not edit sklearn adapter.
- Do not edit dependency files.
- Do not remove existing one-off strategies or mix modes.

## Required Public Configuration
- Extend `SchedulerConfig` with mode `hybrid`.
- Add `hybrid: Optional[Dict[str, Any]] = None`.
- `mode="hybrid"` must require a hybrid config mapping.
- Keep old configs backward-compatible.

## Required Hybrid Config Shape
Support these fields with validation/defaults:
- `mode`: one of:
  - `weighted`
  - `uncertainty_prefilter_diversity`
  - `diversity_prefilter_uncertainty`
- `uncertainty`: one of `entropy`, `margin`, `least_confidence`
- `diversity`: one of `coreset_kcenter`, `embedding_kmeans_pp`, `max_min_embedding`
- `uncertainty_weight`: non-negative float, default `0.5`
- `diversity_weight`: non-negative float, default `0.5`
- `prefilter_multiplier`: positive float, default `3.0`
- `exploration_fraction`: float in `[0, 1]`, default `0.0`
- `class_balance`: bool, default `False`
- `group_balance`: bool, default `False`

## Required Behavior
- Hybrid strategies must fail fast through capability validation:
  - uncertainty component requires `predict_proba`;
  - diversity component requires `embed`.
- Unknown component names or invalid weights/config values must raise `ConfigurationError`.
- Score normalization must be robust:
  - constant scores normalize to zeros;
  - NaN/inf/non-finite scores are handled safely or rejected clearly;
  - extreme finite values do not overflow.
- `weighted` mode:
  - compute normalized uncertainty score and normalized diversity score for all pool items;
  - combine with configured weights;
  - optional `exploration_fraction` reserves deterministic random picks.
- `uncertainty_prefilter_diversity`:
  - rank by uncertainty;
  - keep `ceil(k * prefilter_multiplier)` candidates;
  - rerank/select by diversity.
- `diversity_prefilter_uncertainty`:
  - rank/select candidate set by diversity;
  - rerank by uncertainty.
- `class_balance` and `group_balance` should act as deterministic guardrails when possible:
  - class balance may round-robin predicted class buckets;
  - group balance should avoid selecting repeated groups until groups are exhausted;
  - do not break when group ids are missing.
- Selection must be deterministic, duplicate-free, and only return ids from the pool.
- Edge cases:
  - empty pool;
  - `k <= 0`;
  - `k > pool`;
  - duplicate pool ids;
  - identical scores/embeddings.

## Implementation Guidance
- Prefer placing normalization/scoring helpers in `strategies/hybrid.py`.
- Do not duplicate huge existing strategy code blindly, but it is acceptable to reuse simple entropy/margin/least-confidence and embedding-distance helpers.
- Keep `StrategyScheduler.select_batch()` responsible for mode dispatch and snapshots.
- Hybrid scheduler snapshots should include:
  - `mode`;
  - hybrid config after defaults;
  - selected strategy component names;
  - fallback/exploration counts if used.

## Test Requirements
Add tests covering:
- `SchedulerConfig(mode="hybrid")` validation.
- Missing `predict_proba` or `embed` fails at configure for hybrid.
- Weighted hybrid returns deterministic valid ids on hand-built probabilities/embeddings.
- Prefilter modes behave differently on a small designed geometry.
- Normalization handles constant scores and extreme finite values.
- NaN/inf score handling is tested.
- `class_balance` and `group_balance` guardrails reduce monopolization in a small fake dataset.
- Existing `mix` and `mix_interleaved` tests keep passing.

## Validation
- `uv run --group dev pytest -q tests/test_hybrid_strategy_framework.py tests/test_strategy_capabilities.py tests/test_mix_interleaved_scheduler.py`
- `uv run --group dev pytest -q`

## Forbidden Actions
- Do not run destructive git commands.
- Do not modify benchmark result artifacts.
- Do not revert unrelated dirty worktree changes.

## Acceptance Criteria
- Full tests pass.
- `SchedulerConfig(mode="hybrid", hybrid=...)` works through `StrategyScheduler`.
- Hybrid capability validation is explicit and fail-fast.
