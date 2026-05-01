# 2026-04-27-w110 Exhaustive strategy/cache objections

## Context

The user wants all practical code-review objections to fix at once. Prior audit found hybrid label-width, stale prediction cache, and no-cache embed validation defects. This subtask should enumerate the remaining strategy/cache/model-adapter objections.

## Goal

Audit sampling strategies, scheduler modes, capability inspection, cache semantics, adapter contracts, determinism, validation consistency, and algorithmic edge cases.

## Ownership

Read all files. Do not edit implementation. Do not add files unless absolutely necessary; prefer final findings. Current dirty worktree is user-owned.

## In Scope

- `src/active_learning_sdk/strategies/`
- `src/active_learning_sdk/cache.py`
- `src/active_learning_sdk/adapters/`
- strategy-facing `SelectionContext`/`StrategyScheduler` in `engine.py`
- strategy/cache tests

## Out Of Scope

- Backend HTTP details
- Benchmark report wording except where benchmark relies on strategy behavior
- Implementing fixes

## Required Output

Return a categorized list of objections with severity, file/line references where possible, and evidence type. Include algorithmic risks, contract inconsistencies, and maintainability concerns that should be fixed or documented.
