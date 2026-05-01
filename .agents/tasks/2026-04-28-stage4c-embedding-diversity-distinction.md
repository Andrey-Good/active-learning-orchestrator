# Stage 4C: Embedding Diversity Distinction

## Context

Stage 4A found that `embedding_kmeans_pp` and `max_min_embedding` currently call the same helper with the same initialization, so they are effectively aliases. Hybrid diversity helpers also collapse advertised diversity components too much.

## Goal

Make advertised embedding/diversity strategy names meaningfully distinct while preserving deterministic behavior and existing public strategy names.

## Ownership

May edit:

- `src/active_learning_sdk/strategies/embedding.py`
- `src/active_learning_sdk/strategies/hybrid.py`
- `tests/test_embedding_strategies.py`
- `tests/test_hybrid_strategy_framework.py`

Must not edit:

- `src/active_learning_sdk/strategies/badge.py`
- `src/active_learning_sdk/strategies/stochastic.py`
- `src/active_learning_sdk/engine.py`
- benchmark scripts
- docs

## In Scope

- `max_min_embedding` should remain pure farthest-first / max-min diversity over the unlabeled pool.
- `embedding_kmeans_pp` should use a deterministic k-means++-style initialization that is not identical to max-min. A practical deterministic interpretation is acceptable, for example first choosing the point closest to the global centroid as a representative seed, then greedily maximizing distance to selected centers.
- `coreset_kcenter` should retain labeled samples as existing centers.
- Hybrid diversity scoring/selection should respect different advertised diversity components enough that designed geometry can distinguish them.
- Add designed-geometry tests that would fail if `embedding_kmeans_pp`, `max_min_embedding`, and `coreset_kcenter` collapse into the same order.

## Out Of Scope

- Do not add stochastic random k-means++ sampling.
- Do not change capability requirements.
- Do not touch scheduler snapshots or benchmark evidence.

## Architectural Constraints

- Strategies must be deterministic for identical inputs.
- Existing validation behavior must remain strict.
- Selection must remain duplicate-free and in-pool.
- Keep complexity readable; do not add large framework abstractions.

## Acceptance Criteria

- Designed geometry tests prove distinct selections for `embedding_kmeans_pp`, `max_min_embedding`, and `coreset_kcenter`.
- Hybrid tests prove diversity components can produce different selections on a controlled fixture.
- Existing embedding/hybrid tests pass.
- Full suite should remain green.

## Suggested Validation

```powershell
uv run pytest tests\test_embedding_strategies.py tests\test_hybrid_strategy_framework.py -q
uv run pytest -q
```

## Dependencies

Can run in parallel with Stage 4B because write scopes do not overlap.
