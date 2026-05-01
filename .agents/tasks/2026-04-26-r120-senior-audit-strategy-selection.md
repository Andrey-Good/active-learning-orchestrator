# 2026-04-26-r120 Senior Audit: Strategy And Selection

## Context

The user requested a senior-level code review and acceptance audit of the active-learning SDK: correctness, real behavior under edge cases, code quality, hacks, garbage, stress tests, a written findings file, and benchmark evidence.

This subtask covers acquisition strategies, scheduler selection contracts, capability checks, duplicate/out-of-pool behavior, and existing audit tests in `tests/test_audit_strategy_edge_cases.py`.

## Goal

Find remaining selection/strategy correctness risks, validate whether current edge-case tests are strong, and report algorithmic or architectural objections.

## Responsibility Boundaries

In scope:

- `src/active_learning_sdk/strategies/*`
- scheduler-related portions of `src/active_learning_sdk/engine.py`
- `tests/test_audit_strategy_edge_cases.py`
- related strategy tests

Out of scope:

- Editing files directly.
- Runtime backend/state review except where strategy selection touches it.
- Benchmark implementation changes except observations about measured strategy behavior.

## Explicit Prohibitions

- Do not modify or delete files.
- Do not revert existing uncommitted changes.
- Do not run destructive commands.
- Do not accept duplicate or out-of-pool selection semantics unless explicitly justified by code and tests.

## Execution Plan

1. Inspect strategy implementations and scheduler validation.
2. Run targeted strategy audit tests if feasible.
3. Probe edge cases mentally or with non-persistent commands: empty pool, duplicate IDs, bad probability rows, custom strategy underselection, capability cache invalidation.
4. Return concise findings with file/line references, test commands, and recommended acceptance additions.

## Acceptance Criteria

- Clear verdict on strategy/selection acceptability.
- Findings are source-grounded and actionable.
- Recommended stress cases are concrete.

## Dependencies

Can run in parallel with runtime and benchmark audit subtasks.
