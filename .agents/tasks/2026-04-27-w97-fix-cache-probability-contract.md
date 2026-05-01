# Task W97-E: Fix Cache and Probability Contract Blockers

## Context
Senior audit found P1 release blockers in cache/probability behavior:
- persistent caches alias models whose adapter has no stable `get_model_id()`;
- poisoned prediction cache entries cannot be evicted/recomputed;
- uncertainty strategies silently normalize malformed probability rows;
- one-column `predict_proba` outputs degrade uncertainty strategies into arbitrary tie-breaking.

## Goal
Implement a narrow, production-quality fix for cache and probability-contract blockers.

## Ownership
Allowed write scope:
- `src/active_learning_sdk/cache.py`
- `src/active_learning_sdk/strategies/uncertainty.py`
- if strictly necessary, only the `SelectionContext` cache/model-id methods in `src/active_learning_sdk/engine.py`
- focused tests in `tests/test_w97_senior_audit_acceptance.py`

Do not edit:
- runtime round state-machine logic outside `SelectionContext`;
- state store, backends, lock, README, benchmarks.

## Requirements
- `PredictionCache` must support deleting a single entry.
- `SelectionContext.predict_proba(...)` must validate cached prediction rows before returning them and evict/recompute invalid cached rows.
- Caches must not persistently alias different adapters that lack a stable model id. Prefer a runtime-unique fallback model id over literal `"unknown"` for cache scoping.
- Uncertainty strategies must reject malformed probability rows rather than normalizing arbitrary counts/logits.
- Probability rows must have at least 2 columns for uncertainty strategies.
- Keep legitimate floating-point tolerance for rows that sum approximately to 1.0.

## Acceptance
- Focused tests for the above pass.
- Existing strategy tests continue passing.
- No broad refactor.
