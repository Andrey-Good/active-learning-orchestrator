# w91 - Audit Core Runtime Stress Tests

## Context

The user requested a senior-level code review that tries to break the SDK, not just inspect style. This subtask focuses on the project runtime, state handling, dataset contracts, adapters, caches, and label backend integration surfaces.

## Goal

Find real correctness and robustness problems in the core SDK runtime and add focused stress/regression tests that demonstrate those problems when possible.

## Responsibility Boundaries

Own only:

- `tests/test_audit_runtime_edge_cases.py`
- read-only inspection of `src/active_learning_sdk/project.py`
- read-only inspection of `src/active_learning_sdk/engine.py`
- read-only inspection of `src/active_learning_sdk/state/**`
- read-only inspection of `src/active_learning_sdk/dataset/**`
- read-only inspection of `src/active_learning_sdk/adapters/**`
- read-only inspection of `src/active_learning_sdk/backends/**`

Do not edit SDK implementation files. Do not edit benchmark code. Do not edit docs except through final notes in the agent response.

## In Scope

- Resume/rebind correctness.
- Dataset ID and fingerprint edge cases.
- Duplicate IDs, empty datasets, missing columns, invalid labels.
- State file safety and corrupted/partial state behavior.
- Backend timeout and annotation polling edge cases.
- Adapter capability validation and poor model outputs.
- Tests that expose actual failures or lock down suspicious behavior.

## Out Of Scope

- Strategy algorithm correctness.
- Benchmark quality gate design.
- Long external integrations requiring Docker or network.
- Fixing product code.

## Files/Areas Must Not Touch

- `src/**`
- `benchmarks/**`
- existing tests, unless a tiny import helper change is absolutely necessary.

## Architectural Constraints

- Tests must be deterministic and fast.
- Prefer simulator or tiny fake objects over external services.
- A failing test is acceptable if it demonstrates a real defect; make the failure reason clear.
- Do not mask failures with broad `pytest.raises(Exception)` unless testing a specific failure class is impossible.

## Special Attention

- Look for places where invalid input silently corrupts state.
- Look for behavior that works only because existing tests use happy-path fixtures.
- Check whether persisted state can be resumed with stale runtime objects.

## Forbidden Actions

- No destructive git operations.
- No dependency upgrades.
- No network downloads.
- No Docker.
- Do not revert user changes.

## High-Level Plan

1. Inspect core runtime paths and existing tests.
2. Identify high-risk edge cases.
3. Add `tests/test_audit_runtime_edge_cases.py` with focused repros.
4. Run that test file and report pass/fail details.
5. Return concise findings with file/line references and changed paths.

## Acceptance Criteria

- New tests either pass as regression coverage or fail for clear product bugs.
- Findings identify concrete code locations.
- The final response separates confirmed bugs from suspicious design debt.

## Expected Tests And Validations

- `uv run pytest tests/test_audit_runtime_edge_cases.py -q`

## Dependencies

None.

## Parallel/Sequential Execution

Can run in parallel with strategy and benchmark audit tasks. Write scope is disjoint.
