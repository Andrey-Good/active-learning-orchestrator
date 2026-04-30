# 2026-04-26-r122 Repeat Acceptance: Runtime And State

## Context

The user says the previous senior-review blockers were fixed and requests a repeat SDK acceptance verdict.

Previous runtime blockers:

- backend `push_round()` task IDs were not validated against selected sample IDs;
- pull could partially mutate state before validating the whole backend payload;
- pull could complete with missing selected-task annotations;
- explicit splits allowed train/val/test overlap.

## Goal

Verify whether these runtime/state blockers are actually fixed, whether the new tests are meaningful, and whether any new runtime blockers remain.

## Responsibility Boundaries

In scope:

- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/configs.py`
- `src/active_learning_sdk/state/*`
- `tests/test_senior_acceptance_blockers.py`
- `tests/test_audit_runtime_edge_cases.py`
- related runtime tests

Out of scope:

- Editing files.
- Strategy algorithm review except where runtime state contracts touch it.
- Benchmark methodology review.

## Prohibitions

- Do not modify files.
- Do not revert user changes.
- Do not run destructive commands.

## Plan

1. Inspect the relevant fixed code paths.
2. Run targeted runtime/blocker tests if feasible.
3. Try to falsify the fix with edge-case reasoning.
4. Return verdict, findings, commands/results, and any residual acceptance risks.

## Acceptance Criteria

- Clear accepted/not accepted runtime verdict.
- Source-grounded findings if any.
- Concrete remaining tests if needed.
