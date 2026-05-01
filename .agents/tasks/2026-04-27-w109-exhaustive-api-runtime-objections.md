# 2026-04-27-w109 Exhaustive API/runtime objections

## Context

The user wants the fullest practical list of objections so the SDK can be fixed in one batch. Prior audit found 5 current acceptance defects; this subtask should go beyond those and enumerate API, runtime, state, lifecycle, and backend contract objections.

## Goal

Produce actionable objections, not fixes, covering public API semantics, state machine safety, reconfiguration/migration, split/fingerprint semantics, backend contracts, lifecycle/status/report consistency, and error taxonomy.

## Ownership

Read all files. Do not edit implementation. Do not add files unless absolutely necessary; prefer final findings. Current dirty worktree is user-owned.

## In Scope

- `src/active_learning_sdk/project.py`
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/state/`
- `src/active_learning_sdk/backends/`
- runtime/backend/state tests and docs

## Out Of Scope

- Strategy algorithm details except where they affect public runtime contracts
- Benchmark harness internals
- Implementing fixes

## Required Output

Return a categorized list of additional objections with severity, file/line references where possible, and whether each is confirmed by test/repro/static inspection. Include existing known blockers only if needed for context, but focus on new or broader repair backlog items.
