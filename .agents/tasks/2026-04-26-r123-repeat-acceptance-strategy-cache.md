# 2026-04-26-r123 Repeat Acceptance: Strategy Scheduler Cache

## Context

The user says the previous senior-review blockers were fixed and requests a repeat SDK acceptance verdict.

Previous strategy/cache blockers:

- bandit mode ignored reward state;
- invalid `predict_proba` rows could poison prediction cache before semantic validation;
- custom selector could not inspect the candidate pool;
- hybrid fallback count was underreported.

## Goal

Verify whether these scheduler/cache blockers are actually fixed, whether tests are meaningful, and whether any new strategy-level acceptance blockers remain.

## Responsibility Boundaries

In scope:

- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/strategies/*`
- `src/active_learning_sdk/cache.py`
- `tests/test_senior_acceptance_blockers.py`
- strategy/cache test files

Out of scope:

- Editing files.
- Backend runtime review except shared `SelectionContext` behavior.
- Benchmark methodology review.

## Prohibitions

- Do not modify files.
- Do not revert user changes.
- Do not run destructive commands.

## Plan

1. Inspect bandit selection, cache validation, custom selector contract, and hybrid snapshots.
2. Run targeted strategy/blocker tests if feasible.
3. Probe whether the fixes only satisfy the exact tests or generalize.
4. Return verdict, findings, commands/results, and remaining risks.

## Acceptance Criteria

- Clear accepted/not accepted strategy/cache verdict.
- Source-grounded findings if any.
- Concrete remaining tests if needed.
