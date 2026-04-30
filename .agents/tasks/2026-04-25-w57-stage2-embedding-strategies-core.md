# W57 - Stage 2 Embedding Strategies Core

## Context
Stage 2 adds real embedding-based diversity methods. Stage 1 already introduced capability contracts, so embedding strategies must require `embed` and fail before a run starts when the model cannot provide embeddings.

## Goal
Implement the core embedding strategy slice: embedding validation, real CoreSet/k-center selection, additional deterministic embedding diversity strategies, and cache key scoping.

## Responsibility Boundaries
This is a tightly coupled core slice. Own the SDK core changes needed for embedding strategies.

## In Scope
- `src/active_learning_sdk/cache.py`
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/strategies/__init__.py`
- `src/active_learning_sdk/strategies/uncertainty.py` if needed to remove/replace the placeholder
- New file `src/active_learning_sdk/strategies/embedding.py`
- New tests in `tests/test_embedding_strategies.py`
- New or updated cache/capability tests if needed

## Out of Scope
- Do not edit benchmarks yet.
- Do not edit README or roadmap docs.
- Do not edit sklearn adapter.
- Do not edit dependency files.
- Do not implement BADGE or stochastic/committee strategies.
- Do not add FAISS or other heavy dependencies.

## Required Strategies
Implement these built-in strategy names:
- `coreset_kcenter`
- `embedding_kmeans_pp`
- `max_min_embedding`
- `deduplicate_near_neighbors`
- `density_weighted_diversity`

## Required Behavior
- All embedding strategies declare `required_capabilities = frozenset({"embed"})`.
- `coreset_kcenter` must no longer be an unsupported placeholder.
- `configure()` and `attach_runtime()` must accept these strategies when the model exposes a real `embed`.
- They must fail fast through existing Stage 1 validation when `embed` is missing.
- Selection must be deterministic for the same pool, embeddings, model id, and labeled ids.
- Selection must return valid sample ids from the pool only, with no duplicates.
- Handle edge cases:
  - empty pool;
  - `k <= 0`;
  - `k > len(pool)`;
  - identical embeddings;
  - duplicate embeddings;
  - malformed embeddings: wrong row count, ragged rows, empty rows, non-numeric, non-finite.
- Prefer NumPy implementation only; no FAISS.
- Use stable tie-breaking, not Python hash randomization.

## Cache Key Contract
Embedding cache keys must include enough scope to avoid cross-dataset embedding reuse:
- model id;
- dataset fingerprint when available;
- embedding config/version string when available;
- sample id.

Implementation guidance:
- It is acceptable to default dataset fingerprint and embedding config to stable values when unavailable.
- `SelectionContext.embed()` should pass the current dataset fingerprint into `EmbeddingCache` when available.
- Preserve backward compatibility for direct `EmbeddingCache.get(model_id, sample_id)` calls if any tests or users rely on it.

## Test Requirements
Add tests covering:
- `coreset_kcenter` chooses diverse endpoints on a simple 1D/2D geometry.
- `coreset_kcenter` uses labeled ids as existing centers when available.
- `embedding_kmeans_pp`, `max_min_embedding`, `deduplicate_near_neighbors`, and `density_weighted_diversity` return deterministic valid diverse batches.
- Identical embeddings still return deterministic unique ids.
- Malformed embeddings raise `ConfigurationError`.
- Configuring `coreset_kcenter` without `embed` fails at `configure()`.
- Configuring `coreset_kcenter` with an embedding-capable model succeeds.
- Embedding cache keys differ by dataset fingerprint.

## Validation
- `uv run --group dev pytest -q tests/test_embedding_strategies.py`
- `uv run --group dev pytest -q`

## Forbidden Actions
- Do not run destructive git commands.
- Do not revert unrelated dirty worktree changes.
- Do not modify accepted benchmark artifacts.

## Acceptance Criteria
- Full tests pass.
- `coreset_kcenter` is a real working strategy.
- The SDK has deterministic embedding diversity methods ready for benchmark diagnostics.
