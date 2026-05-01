# 2026-04-27-w106 Senior runtime/state audit

## Context

The user requested a hard senior-level code review of the current Active Learning SDK, including real stress tests, edge cases, and maintainability findings. The repo already contains prior audit reports and xfail known-defect tests. This task must verify the current runtime/state behavior and identify remaining acceptance blockers without reverting existing work.

## Goal

Find runtime, state-machine, split, resume, backend, and public lifecycle defects that can break real SDK use or make the project hard to maintain.

## Ownership

May read all repo files. May propose findings and minimal test files under `tests/` only if needed. Do not edit implementation files. Do not edit benchmark scripts. Do not alter existing docs except by reporting findings to the orchestrator.

## In Scope

- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/project.py`
- `src/active_learning_sdk/state/`
- `src/active_learning_sdk/backends/`
- runtime-related tests under `tests/`
- existing audit docs and tmp notes relevant to runtime/state

## Out Of Scope

- Strategy scoring algorithm rewrites
- Benchmark harness implementation
- Packaging metadata changes
- Fixing defects

## Constraints

- Treat current dirty worktree as user-owned; do not revert anything.
- Prefer reproducible evidence: exact test names, commands, and observed failures.
- Distinguish confirmed current defects from historical findings already fixed.
- If adding tests, make them narrow and mark known unfixed defects explicitly with `pytest.mark.xfail` plus a reason.

## Execution Plan

1. Inspect runtime/state/backends and related tests.
2. Run focused tests or small repro scripts for suspected issues.
3. Produce concise findings with severity, file/line references where possible, and evidence.

## Acceptance Criteria

- Findings are specific enough to fix.
- At least one focused command verifies the runtime/state surface.
- No implementation files are modified.

## Dependencies

Can run in parallel with strategy/cache and benchmark/docs audits.
