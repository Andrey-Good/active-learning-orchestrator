# Task W97-J: Simulator Resume Task Binding Integrity

## Context
Final independent review rejected release because `SimulatorLabelBackend.restore_round_samples(...)` can recreate tasks from corrupt persisted `task_ids`, accepting swapped or malformed IDs after restart.

## Goal
Fail closed on simulator resume if persisted task IDs do not match the deterministic simulator binding `sim:{round_id}:{sample_id}`.

## Ownership
May change:
- `src/active_learning_sdk/backends/simulator.py`
- `tests/test_w97_runtime_state_backends.py`
- narrowly related simulator/backend tests if needed

Must not change:
- benchmark harnesses
- unrelated backend implementations
- public scheduler/strategy code

## Scope
In scope:
- Validate provided `task_ids` before creating/rebinding simulator tasks on restore.
- Reject swapped, cross-round, malformed, or duplicate task IDs with `LabelBackendError`.
- Add a resume-path test that mutates persisted state task IDs, starts a fresh simulator/backend instance, and proves the SDK rejects the corrupted mapping.

Out of scope:
- Changing deterministic simulator task-id format.
- Adding persistent simulator storage.

## Acceptance Criteria
- Focused W97 runtime/backend tests pass.
- Existing simulator/backend tests pass.
- Full test suite remains green after integration.
