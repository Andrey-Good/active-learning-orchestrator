# w92 - Fix Audit Strategy Contracts

## Context

The audit added failing strategy/scheduler acceptance tests:

- uncertainty strategies can return duplicate IDs when pool contains duplicate IDs;
- random can preserve duplicates when `k >= len(pool_ids)`;
- single/custom scheduler accepts out-of-pool IDs;
- single mode underfills after duplicate strategy output.

## Goal

Enforce a clean selection contract: selected IDs are unique, belong to the current pool, and batch selection fills to `min(k, unique_pool_size)` when possible for normal built-in under-selection after deduplication.

## Ownership

You own:

- `src/active_learning_sdk/engine.py` scheduler-related selection helpers only
- `src/active_learning_sdk/strategies/uncertainty.py`
- strategy audit tests only if needed, primarily `tests/test_audit_strategy_edge_cases.py`

Coordinate mentally with w91: do not edit runtime configure/label/cache validation areas.

## In Scope

- Deduplicate candidate pool IDs before scoring in built-in uncertainty/random strategies.
- Make `RandomStrategy` return unique IDs capped to `k`.
- Validate scheduler outputs against the candidate pool in every scheduler mode.
- Fail fast on out-of-pool IDs from custom/single strategy output.
- Refill deterministically from remaining pool IDs after duplicate/short output where appropriate.

## Out Of Scope

- No backend/runtime label validation.
- No benchmark changes.
- No README changes.

## Acceptance Criteria

- `uv run pytest tests/test_audit_strategy_edge_cases.py -q` passes.
- Existing strategy tests still pass.
- No selected ID outside the input pool can pass through `StrategyScheduler.select_batch`.

## Notes

Do not silently hide broken out-of-pool selectors. Raise `ConfigurationError` with a clear message.
