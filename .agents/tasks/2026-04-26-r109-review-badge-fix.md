# Task r109: Review BADGE Fix

## Context
The orchestrator implemented BADGE improvements:
- `BadgeStrategy` now uses a non-oracle cold-start guardrail: sparse label probability support triggers embedding novelty via `KCenterGreedyStrategy`, with deterministic random fallback.
- The benchmark sklearn BADGE proxy now hashes sparse TF-IDF gradient residuals into a bounded 1024-dimensional vector instead of materializing dense `num_labels * vocab` vectors.
- Tests were added for BADGE cold-start fallback and bounded deterministic benchmark gradient embeddings.

## Goal
Review the BADGE patch for correctness, architecture, non-leakage, determinism, and test adequacy.

## Responsibility Boundaries
Read-only review. Do not edit files.

## In Scope
- `src/active_learning_sdk/strategies/badge.py`
- `benchmarks/sdk_first_benchmark.py`
- `tests/test_badge_strategy.py`
- `tests/test_sdk_first_benchmark_embedding_diagnostics.py`

## Out of Scope
- Do not review unrelated dirty worktree changes.
- Do not propose oracle-label-based fixes.

## Review Questions
- Does BADGE cold-start fallback avoid oracle leakage?
- Does the optional probability-support check preserve BADGE's required capability contract?
- Is swallowing optional `predict_proba` failures acceptable here?
- Is the hashed gradient proxy deterministic and bounded?
- Are tests sufficient to prevent regression?

## Acceptance Criteria
- Final answer lists findings with severity and file/line references.
- If no blocking findings, state residual risks.

## Dependencies
Depends on current local BADGE patch.
