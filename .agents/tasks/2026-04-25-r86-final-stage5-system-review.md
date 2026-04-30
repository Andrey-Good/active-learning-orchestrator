# R86 - Final Stage 5 System Review

## Context
Stage 5 goal: configurable hybrid strategy framework.

Completed subtasks:
- W66 implemented hybrid framework core.
- R84 reviewed hybrid core with no findings.
- W67 wired hybrid strategies into benchmark harness.
- R85 reviewed hybrid benchmark wiring with no findings.

## Goal
Perform final Stage 5 system review and decide whether Stage 5 can close.

## Responsibility Boundaries
- This is a read-only system review.
- Review Stage 5 as an integrated product slice.

## In Scope
- Core:
  - `src/active_learning_sdk/configs.py`
  - `src/active_learning_sdk/engine.py`
  - `src/active_learning_sdk/strategies/hybrid.py`
  - `src/active_learning_sdk/strategies/__init__.py`
- Tests:
  - `tests/test_hybrid_strategy_framework.py`
  - `tests/test_strategy_capabilities.py`
  - `tests/test_sdk_first_benchmark_embedding_diagnostics.py`
- Benchmark:
  - `benchmarks/sdk_first_benchmark.py`

## Out of Scope
- Do not edit files.
- Do not implement Stage 6.
- Do not regenerate committed benchmark artifacts.

## Review Questions
- Does `SchedulerConfig(mode="hybrid", hybrid=...)` work and validate safely?
- Do hybrid modes cover weighted and prefilter/rerank composition?
- Are score normalization and guardrails tested?
- Do capability checks fail fast for missing `predict_proba`/`embed`?
- Are representative benchmark presets available and smoke-tested?
- Does full suite pass?

## Validation
- `uv run --group dev pytest -q`
- `uv run python benchmarks/sdk_first_benchmark.py --preset smoke --strategies random,hybrid_weighted_entropy_coreset --datasets grouped_duplicates --budgets 12 --seeds 13 --output-dir <temp path>`
- `uv run python benchmarks/sdk_first_benchmark.py --preset smoke --strategies random,hybrid_weighted_guarded --datasets grouped_duplicates --budgets 12 --seeds 13 --output-dir <temp path>`

## Acceptance Criteria
- No open blockers remain for Stage 5.
- Reviewer explicitly says Stage 5 can close if clean.
