# W99 Strategy Correctness Acceptance Audit

## Context
The user requested a senior-level review that tries to break the SDK, especially hidden algorithmic and edge-case problems. This subtask covers strategy selection correctness, probability validation, cache-sensitive behavior, duplicate IDs, pathological inputs, and deterministic tie handling.

## Goal
Find concrete strategy defects or fragile behavior and add focused tests that demonstrate real problems or lock down important behavior.

## Responsibility Boundaries
Owned write scope:
- `tests/test_acceptance_strategy_correctness_2026_04_27.py`
- optional notes in `.agents/tmp/w99-strategy-findings.md`

Read scope:
- `src/active_learning_sdk/strategies/**`
- `src/active_learning_sdk/engine.py` selection context and scheduler sections
- `src/active_learning_sdk/cache.py`
- existing strategy tests under `tests/`

Do not modify production SDK code, benchmark code, docs, pyproject, lock files, or other tests.

## In Scope
- Probability shape/value validation.
- Duplicate pool IDs and non-string ID behavior.
- Cache invalidation/stale prediction risks.
- Strategy parity against direct formulas where practical.
- Determinism under ties and repeated calls.

## Out of Scope
- Fixing discovered SDK defects.
- Runtime orchestration tests.
- External library benchmarking.
- Installing new dependencies.

## Constraints
- Keep tests local-only, deterministic, and fast.
- Do not use network, Docker, or external datasets.
- Do not write generated artifacts outside the owned test file and optional `.agents/tmp` note.

## Execution Plan
1. Inspect strategy implementations and existing tests.
2. Pick high-risk edge cases with small fixtures.
3. Add pytest coverage in the owned file.
4. Run the new tests and relevant nearby strategy tests.
5. Record confirmed issues, non-issues, and validation output in `.agents/tmp/w99-strategy-findings.md`.

## Acceptance Criteria
- At least one meaningful strategy stress/regression test is added.
- Tests expose or guard behavior that matters for SDK correctness.
- Findings include exact file/function references and validation status.

## Dependencies
Can run in parallel with runtime and benchmark subtasks. No write-scope overlap.
