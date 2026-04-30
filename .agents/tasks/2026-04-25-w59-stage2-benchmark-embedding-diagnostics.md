# W59 - Stage 2 Benchmark Embedding Diagnostics

## Context
Stage 2 core embedding strategies exist. The benchmark harness must now be able to evaluate them with relevant diversity diagnostics instead of only quality metrics.

## Goal
Extend the SDK-first benchmark harness to run embedding strategies and report diagnostics that show whether embedding diversity reduces redundancy and group concentration.

## Responsibility Boundaries
Own benchmark diagnostics only.

## In Scope
- `benchmarks/sdk_first_benchmark.py`
- New or updated tests under `tests/` for benchmark strategy availability and diagnostics.

## Out of Scope
- Do not edit SDK core strategy implementation.
- Do not edit README or roadmap docs.
- Do not regenerate committed benchmark artifacts.
- Do not edit dependency files.

## Required Benchmark Behavior
- The scheduler-level benchmark adapter must expose a deterministic `embed(texts, batch_size=...)` capability so embedding strategies can run.
- Embeddings should be cheap, deterministic, and meaningful enough for synthetic text:
  - TF-IDF vectors from the fitted benchmark pipeline are acceptable;
  - if model is unfitted, fail clearly.
- `strategy_specs()` must include:
  - `coreset_kcenter`
  - `embedding_kmeans_pp`
  - `max_min_embedding`
  - `deduplicate_near_neighbors`
  - `density_weighted_diversity`
- Smoke preset does not have to include every embedding method by default, but full preset should be able to run them.
- Selection diagnostics must include:
  - `selected_duplicate_rate`;
  - nearest-neighbor redundancy, for example mean/min nearest-neighbor distance among selected items when embeddings are available;
  - existing `group_hhi` and `top_group_fraction` must remain.
- Diagnostics must be strict-JSON safe.

## Tests
Add tests that:
- `strategy_specs()` contains all five embedding strategies.
- `compute_selection_diagnostics` or the benchmark curve path emits the new redundancy metrics.
- A tiny benchmark curve with an embedding strategy completes without external services or downloads.

## Validation
- Run the new/updated tests.
- Run `uv run python benchmarks/sdk_first_benchmark.py --preset smoke --strategies random,coreset_kcenter --datasets grouped_duplicates --budgets 12 --seeds 13 --output-dir <temp path>`.
- Run `uv run --group dev pytest -q`.

## Files That Must Not Be Touched
- `src/**`
- `README.md`
- `benchmarks/results/**`
- `pyproject.toml`
- `uv.lock`

## Forbidden Actions
- Do not run destructive git commands.
- Do not modify accepted benchmark artifacts.
- Do not revert unrelated changes.

## Acceptance Criteria
- Embedding strategies are benchmarkable.
- Benchmark artifacts include redundancy metrics needed to compare diversity methods.
- Full tests pass.
