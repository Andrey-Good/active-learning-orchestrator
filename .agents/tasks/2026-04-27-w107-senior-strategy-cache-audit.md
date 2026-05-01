# 2026-04-27-w107 Senior strategy/cache audit

## Context

The user requested aggressive review of code quality, correctness, hidden bugs, dirty generated-code patterns, and stress behavior. This subtask focuses on acquisition strategies, model capability contracts, prediction/embedding cache semantics, and edge-case validation.

## Goal

Find correctness and maintainability defects in strategy selection and cache behavior, especially cases where invalid model output, stale cache entries, tie handling, label-schema mismatch, or optional capability drift changes acquisition results.

## Ownership

May read all repo files. May propose/add focused tests under `tests/` only if necessary. Do not edit implementation files. Do not edit benchmark scripts or docs directly.

## In Scope

- `src/active_learning_sdk/cache.py`
- `src/active_learning_sdk/strategies/`
- `src/active_learning_sdk/adapters/`
- `SelectionContext` strategy-facing methods in `src/active_learning_sdk/engine.py`
- existing strategy/cache tests

## Out Of Scope

- Runtime state-machine lifecycle outside strategy-facing behavior
- External benchmark design
- Fixing defects

## Constraints

- Current dirty tree is user-owned; do not revert.
- Stress invalid probabilities, malformed embeddings, duplicate IDs, inconsistent dimensions, cache-disabled vs cache-enabled parity.
- Prefer regression tests that fail with `--runxfail` if documenting known unfixed behavior.

## Execution Plan

1. Inspect strategy contracts and cache implementation.
2. Run focused current tests and targeted repros.
3. Return findings with severity, evidence, and recommended acceptance tests.

## Acceptance Criteria

- Findings separate real correctness bugs from style complaints.
- Commands and repro snippets are concrete.
- No implementation files are modified.

## Dependencies

Can run in parallel with runtime/state and benchmark/docs audits.
