# W98 Runtime/State Acceptance Audit

## Context
The user requested a senior-level acceptance review of the SDK, focused on real correctness, maintainability, hidden failure modes, and reproducible evidence. This subtask covers runtime orchestration, state persistence, resume semantics, locking, public API behavior, and backend boundary safety.

## Goal
Find concrete runtime/state/API defects or maintenance blockers and add focused regression/stress tests that demonstrate real problems or protect critical behavior.

## Responsibility Boundaries
Owned write scope:
- `tests/test_acceptance_runtime_state_2026_04_27.py`
- optional notes in `.agents/tmp/w98-runtime-state-findings.md`

Read scope:
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/project.py`
- `src/active_learning_sdk/state/**`
- `src/active_learning_sdk/backends/**`
- existing tests under `tests/`

Do not modify production SDK code, benchmark code, docs, pyproject, lock files, or other tests.

## In Scope
- Resume/idempotency edge cases.
- State corruption or dataset mismatch cases.
- Locking and state-store safety issues.
- Public API behavior that can silently do the wrong thing.
- Tests should be deterministic and local-only.

## Out of Scope
- Fixing discovered SDK defects.
- Installing new dependencies.
- Benchmark implementation.
- Strategy algorithm changes.

## Constraints
- Do not delete or revert existing user changes.
- Set up tests so they do not require network, Docker, Label Studio, or external services.
- Avoid writing generated artifacts outside the owned test file and optional `.agents/tmp` note.
- Use existing local helper patterns where possible.

## Execution Plan
1. Inspect runtime/state code and nearby tests.
2. Identify high-signal edge cases that can be reproduced quickly.
3. Add focused pytest tests in the owned file.
4. Run the new tests and relevant nearby tests.
5. Write concise findings and validation notes to `.agents/tmp/w98-runtime-state-findings.md`.

## Acceptance Criteria
- At least one meaningful runtime/state/API stress or regression test is added.
- Tests are deterministic and isolated.
- Findings clearly distinguish confirmed failures from residual risks.
- Validation commands and outcomes are recorded.

## Dependencies
Can run in parallel with strategy and benchmark subtasks. No write-scope overlap.
