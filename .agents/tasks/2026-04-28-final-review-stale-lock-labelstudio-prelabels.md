# Task W113 Review: stale-lock and Label Studio direct prelabel fixes

## Context

This is a final reviewer pass after remediation of two remaining P1 blockers from the repeat senior acceptance review:

- stale project-lock cleanup could unlink a fresh replacement lock in a stale-break race;
- `LabelStudioBackend.push_round()` accepted direct probability prelabels such as `[0.6, 0.6]` and imported them as predictions.

## Goal

Confirm whether the current implementation is acceptable for these two blockers and whether the changes introduced regressions in adjacent runtime, backend, or public-contract behavior.

## Responsibility Boundaries

Review only. Do not edit files.

## In Scope

- `src/active_learning_sdk/state/lock.py`
- `src/active_learning_sdk/backends/label_studio.py`
- related tests in:
  - `tests/test_deep_audit_runtime_state_backends_2026_04_28.py`
  - `tests/test_w97_runtime_state_backends.py`
  - `tests/test_label_backends.py`
  - `tests/test_objection_sweep_security_infra_2026_04_28.py`
- README/docs status only insofar as they claim current validation evidence.

## Out Of Scope

- unrelated strategy quality work;
- benchmarks;
- broad API redesign;
- formatting-only issues unless they hide a correctness problem.

## Architectural Constraints

- Project lock must remain cross-platform and must not delete a lock file it does not own or that changed during stale cleanup.
- Stale dead-process locks must still be recoverable.
- Label Studio direct prelabels must not import invalid probability rows.
- Valid direct label prelabels and valid probability rows must still work.

## Forbidden Actions

- Do not modify production code or tests.
- Do not mask failures by changing assertions.
- Do not run destructive git commands.

## Suggested Validation

- `uv run pytest tests\test_deep_audit_runtime_state_backends_2026_04_28.py tests\test_w97_runtime_state_backends.py -q`
- `uv run pytest tests\test_label_backends.py tests\test_objection_sweep_security_infra_2026_04_28.py -q`
- If time allows, inspect whether `ProjectLock.acquire()` has any new deadlock or stale gate failure mode.

## Acceptance Criteria

- No remaining P1/P2 blocker in the two remediated areas.
- Any concern must include concrete file/line references and a reproducing scenario.
- If no blockers remain, explicitly state that the remediation is accepted for this scope.
