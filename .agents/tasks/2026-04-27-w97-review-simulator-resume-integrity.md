# Task W97-L: Review Simulator Resume Integrity P1 Fix

## Context
Task W97-J changed `src/active_learning_sdk/backends/simulator.py` and `tests/test_w97_runtime_state_backends.py` after final review found a P1 blocker:
- simulator resume could recreate tasks from corrupt persisted task IDs and accept swapped bindings.

## Goal
Perform a read-only senior review of the W97-J fix.

## Scope
Read only:
- `src/active_learning_sdk/backends/simulator.py`
- `tests/test_w97_runtime_state_backends.py`
- runtime resume path in `src/active_learning_sdk/engine.py` if needed to understand call flow

Do not edit files.

## Review Questions
- Does `restore_round_samples(...)` validate persisted task IDs before creating/rebinding tasks?
- Are swapped, malformed, duplicate, cross-round, and missing persisted task IDs rejected?
- Does the resume-path test exercise a fresh backend instance and corrupted persisted state?
- Does the change preserve normal simulator push/poll/pull behavior?
- Are errors fail-closed and useful?

## Output
Return findings ordered by severity. If no release blockers remain in this scope, say so clearly and list residual risks.
