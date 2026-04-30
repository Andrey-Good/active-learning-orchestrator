# 2026-04-27-w96 Senior Audit Runtime/State

## Context

Part of the senior acceptance audit requested on 2026-04-27. The goal is to attack the SDK as if accepting externally produced code: find correctness, durability, and maintainability defects that would make the repository hard to keep working over one or two years.

## Goal

Audit runtime orchestration, persistence, resume safety, label/backend state transitions, and public project lifecycle. Add focused acceptance/stress tests that expose real defects if found, and record precise findings for the final audit document.

## Responsibility Boundaries

May inspect all repository files. May change only:

- `tests/test_senior_audit_runtime_state_2026_04_27.py`
- `.agents/tmp/2026-04-27-w96-runtime-state-notes.md`

Must not edit SDK implementation files, benchmark scripts, README, or final docs.

## In Scope

- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/project.py`
- `src/active_learning_sdk/state/*`
- `src/active_learning_sdk/backends/*`
- runtime tests already in `tests/test_*runtime*`, `tests/test_state_safety.py`, `tests/test_label_backends.py`
- edge cases: interrupted rounds, malformed state, stale runtime attachments, backend contract violations, non-finite metrics, duplicate/overlapping ids, budget semantics.

## Out of Scope

- Strategy quality/formula audit except when it affects runtime state.
- Benchmark harness changes.
- Fixing production code.

## Constraints

- Do not revert existing user changes.
- Tests should be plain pytest tests. If a test demonstrates a current defect, leave it as a normal failing acceptance test unless the failure would make collection impossible.
- Prefer small deterministic fixtures over network, Docker, or external services.
- Record every finding with file/line, expected behavior, observed behavior, and reproduction command.

## Execution Plan

1. Inspect runtime/state/backend implementation and existing tests.
2. Identify high-value defect candidates.
3. Add targeted pytest tests in the owned test file.
4. Run the new tests and relevant existing runtime tests.
5. Write notes to `.agents/tmp/2026-04-27-w96-runtime-state-notes.md`.

## Acceptance Criteria

- New test file exists and is runnable.
- Notes identify confirmed defects or explicitly say no defect was confirmed in this area.
- Notes include commands run and exact pass/fail outcomes.

## Dependencies

Runs in parallel with strategy and benchmark audit tasks. No shared write scope.
