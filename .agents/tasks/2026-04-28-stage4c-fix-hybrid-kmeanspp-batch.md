# Stage 4C Fix: Hybrid KMeans++ Batch Semantics

## Context

Stage 4B/4C reviewer rejected the strategy-quality fix because hybrid weighted `embedding_kmeans_pp` applies the new definition only to the first pick. For `k > 1`, it ranks by static centroid-representative score instead of the advertised deterministic k-means++-style greedy farthest-prototype progression.

## Goal

Fix hybrid weighted diversity so `embedding_kmeans_pp` batch selection is meaningfully greedy for `k > 1`, and add tests that fail on the rejected static-ranking behavior.

## Ownership

May edit:

- `src/active_learning_sdk/strategies/hybrid.py`
- `tests/test_hybrid_strategy_framework.py`

Must not edit:

- `src/active_learning_sdk/strategies/embedding.py`
- `src/active_learning_sdk/strategies/badge.py`
- `src/active_learning_sdk/strategies/stochastic.py`
- `src/active_learning_sdk/engine.py`
- benchmark scripts

## In Scope

- Weighted hybrid with pure diversity (`uncertainty_weight=0`, `diversity_weight=1`) should use batch-level greedy order for `embedding_kmeans_pp`, not a static score sort.
- It is acceptable to use `_diversity_select(...)` order for diversity-dominant weighted hybrid paths when diversity weight is positive and uncertainty weight is zero.
- Add a controlled `k > 1` test where static centroid-representative ranking would pick the wrong second item.

## Out Of Scope

- Do not redesign all weighted-hybrid multi-objective ranking.
- Do not change public config names.
- Do not add benchmark scripts in this fix.

## Acceptance Criteria

- New test catches the exact reviewer finding.
- Focused hybrid tests pass.
- Stage 4B/4C focused suite passes.
- Full suite should remain green.

## Suggested Validation

```powershell
uv run pytest tests\test_hybrid_strategy_framework.py -q
uv run pytest tests\test_badge_strategy.py tests\test_stochastic_committee_strategies.py tests\test_embedding_strategies.py tests\test_hybrid_strategy_framework.py -q
uv run pytest -q
```
