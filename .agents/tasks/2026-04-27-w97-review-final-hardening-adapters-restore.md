# Task W97-O: Review Final Hardening For Adapter Probability And Restore Validation

## Context
Task W97-N closed residual findings from the final system review:
- sklearn adapter normalized invalid `predict_proba` rows;
- restore fallback caught internal `TypeError`;
- WAIT path lacked generic task-id validation before poll.

## Goal
Perform a read-only senior review of the W97-N hardening.

## Scope
Read only:
- `src/active_learning_sdk/adapters/sklearn.py`
- `src/active_learning_sdk/engine.py`
- `tests/test_sklearn_adapter.py`
- `tests/test_w97_runtime_state_backends.py`

Do not edit files.

## Known Validation
- `uv run pytest tests/test_sklearn_adapter.py tests/test_w97_runtime_state_backends.py tests/test_reference_strategy_benchmark.py tests/test_project_smoke_benchmark.py -q` -> `43 passed`
- `uv run pytest -q` -> `354 passed`
- `uv build` -> success

## Review Questions
- Does the sklearn adapter now reject count-like probability rows rather than renormalizing them?
- Does strict adapter validation preserve legitimate sklearn and decision-function behavior?
- Does restore call-style dispatch preserve legacy two-argument backends without swallowing internal `TypeError`s?
- Does WAIT validate task-id mappings before backend poll?
- Are the new tests meaningful and enough for regression protection?

## Output
Return findings ordered by severity. If no release blockers or meaningful residual findings remain in this scope, say so clearly.
