# Task W101: Hard Audit Runtime And State

## Context

The user requested a senior-level acceptance audit of the SDK quality, correctness, hidden failure modes, dirty code, shortcuts, and maintainability risks. This subtask owns runtime orchestration, state persistence, backend lifecycle, lock/state integrity, and public project lifecycle behavior.

## Goal

Find concrete defects or serious risks in runtime/state behavior. Prefer evidence that can be reproduced with tests or command output. This is an audit task, not a broad refactor.

## Responsibility Boundaries

May inspect:

- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/project.py`
- `src/active_learning_sdk/state/**`
- `src/active_learning_sdk/backends/**`
- relevant runtime tests under `tests/`

May write only:

- `.agents/tmp/w101-runtime-state-findings.md`

Do not edit source code or tests unless explicitly redirected by the orchestrator.

## In Scope

- Resume/idempotency failures
- State corruption, partial writes, stale locks, atomicity gaps
- Backend push/pull/task binding correctness
- Timeout and stop-criteria edge cases
- Public API behavior that contradicts README claims
- Minimal repro snippets or test ideas for each issue

## Out Of Scope

- Strategy math and cache selection correctness, except where it impacts runtime state
- Benchmark quality claims
- Documentation polish unrelated to runtime correctness

## Architectural Constraints

- Do not assume Label Studio is available locally.
- Prefer existing simulator/backend tests for reproducible evidence.
- Work with the dirty worktree as-is; do not revert or normalize existing changes.

## Forbidden Actions

- Do not run destructive commands.
- Do not install persistent dependencies.
- Do not modify `uv.lock`, `pyproject.toml`, source files, or tests.
- Do not write benchmark result directories.

## Execution Plan

1. Read the runtime/state/backend code and tests.
2. Identify high-risk contracts and compare them to README promises.
3. Run narrow tests only if needed and feasible.
4. Write findings to `.agents/tmp/w101-runtime-state-findings.md` with severity, evidence, and repro guidance.

## Acceptance Criteria

- Findings are concrete, file/line grounded, and not generic style complaints.
- Each serious issue includes a reproducible scenario or a precise missing test.
- If no blocker is found, explicitly state residual risks and why.

## Dependencies

Can run in parallel with W102 and W103. No write-scope overlap.
