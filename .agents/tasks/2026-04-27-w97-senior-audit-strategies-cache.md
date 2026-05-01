# 2026-04-27-w97 Senior Audit Strategies/Cache

## Context

Part of the senior acceptance audit requested on 2026-04-27. The user wants the code stressed for hidden bugs, dirty shortcuts, and behavior that will not survive long-term maintenance.

## Goal

Audit selection strategies, scheduler behavior, prediction/embedding cache semantics, and model capability enforcement. Add focused acceptance/stress tests that expose real defects if found, and record precise findings for the final audit document.

## Responsibility Boundaries

May inspect all repository files. May change only:

- `tests/test_senior_audit_strategies_cache_2026_04_27.py`
- `.agents/tmp/2026-04-27-w97-strategies-cache-notes.md`

Must not edit SDK implementation files, benchmark scripts, README, or final docs.

## In Scope

- `src/active_learning_sdk/strategies/*`
- scheduler logic in `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/cache.py`
- adapter capability contracts in `src/active_learning_sdk/adapters/*`
- edge cases: duplicate ids, malformed probabilities/embeddings, cache key collisions, cold-start fallback behavior, mix/hybrid allocation, custom selector contracts.

## Out of Scope

- Backend/persistence audit except where cache state affects selection.
- Benchmark script implementation.
- Fixing production code.

## Constraints

- Do not revert existing user changes.
- Tests should be deterministic and avoid external downloads/services.
- If a confirmed current defect is found, express it as a normal failing pytest acceptance test.
- Avoid overlapping test helper names with other owned files unless local to the file.

## Execution Plan

1. Inspect strategies, scheduler, cache, and existing strategy tests.
2. Look for contract mismatches and edge cases absent from tests.
3. Add targeted pytest tests in the owned test file.
4. Run the new tests and relevant existing strategy/cache tests.
5. Write notes to `.agents/tmp/2026-04-27-w97-strategies-cache-notes.md`.

## Acceptance Criteria

- New test file exists and is runnable.
- Notes contain confirmed findings or a no-finding statement.
- Commands and outcomes are included.

## Dependencies

Runs in parallel with runtime and benchmark audit tasks. No shared write scope.
