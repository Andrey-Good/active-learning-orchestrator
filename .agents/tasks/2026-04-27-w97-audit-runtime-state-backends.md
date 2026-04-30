# Task W97-A: Runtime, State, Backends Senior Audit

## Context
The user asked for a hard senior-level acceptance review of the SDK: find hidden defects, fragile logic, AI-generated code smell, missing edge cases, and release-blocking maintainability risks. This is the first audit pass before any new fixes.

## Goal
Perform a read-only audit of runtime orchestration, persistent state, label backends, public project API, and resume/idempotency behavior. Return concrete findings that can become tests or fixes.

## Ownership
Read-only scope:
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/project.py`
- `src/active_learning_sdk/state/**`
- `src/active_learning_sdk/backends/**`
- Runtime/backend/state tests under `tests/`

Do not edit files.

## In Scope
- State loading/saving and corruption checks.
- Round state-machine transitions and idempotency.
- Backend `push/poll/pull` contracts.
- Public API misuse and edge cases.
- Resume behavior after process restart.
- JSON serialization and runtime-only object boundaries.

## Out of Scope
- Strategy math details, except where engine calls strategies incorrectly.
- Benchmark implementation.
- Documentation rewrites.

## Special Attention
- Look for partial-state mutation before validation.
- Look for silently accepted corrupt state.
- Look for task/sample ID mismatches.
- Look for backends that cannot resume or violate their own contract.
- Look for functions that claim public support but only work in toy paths.

## Expected Output
- Findings ordered by severity.
- For each finding: file/line, reproduction idea, likely failing test name, and whether it is release-blocking.
- If no release blockers, say so explicitly and list residual risks.
