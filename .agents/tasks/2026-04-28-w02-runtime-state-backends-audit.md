# 2026-04-28-w02-runtime-state-backends-audit

## Context

Part of a senior acceptance audit requested on 2026-04-28. The target is runtime correctness under edge cases, state persistence, locking, annotation orchestration, and backend contracts.

## Goal

Find defects that would break real long-lived SDK use: resume behavior, duplicate IDs, corrupt state, concurrent access, backend failure handling, timeout semantics, and resource cleanup. Produce focused tests plus findings.

## Responsibility Boundaries

Owner may change only:

- `tests/test_deep_audit_runtime_state_backends_2026_04_28.py`
- `.agents/tmp/2026-04-28-w02-runtime-state-backends-findings.md`

Owner must not change:

- `src/**`
- existing `tests/**`
- `benchmarks/**`
- docs other than the owned findings file

## In Scope

- `engine.py`, `project.py`, `annotation.py`
- `state/store.py`, `state/lock.py`
- `backends/base.py`, `backends/simulator.py`, `backends/label_studio.py`, `backends/managed_docker.py`
- Cache interactions only when tied to runtime correctness.

## Out of Scope

- Strategy ranking math unless it causes runtime failure.
- Public API packaging.
- Reference benchmark comparisons.

## Constraints

- Do not require Docker or a live Label Studio server.
- Mock external/process boundaries where needed.
- Keep tests deterministic and bounded in runtime.

## Execution Plan

1. Map runtime lifecycle and persisted state schema.
2. Attack edge cases: empty pools, repeated labels, partial/corrupt state, interrupted rounds, duplicate selection IDs, timeout/failure callbacks, concurrent lock acquisition.
3. Add tests that demonstrate concrete broken behavior.
4. Write findings with reproduction commands and expected vs actual behavior.

## Acceptance Criteria

- New tests expose at least one meaningful runtime/state/backend defect if found.
- Findings are specific enough for a fixer to implement without guessing.
- No production code is edited.

## Validation

- Run the new test file directly.
- Run selected existing runtime/state/backend tests if time permits.

## Dependencies

Can run in parallel with W01, W03, and W04.
