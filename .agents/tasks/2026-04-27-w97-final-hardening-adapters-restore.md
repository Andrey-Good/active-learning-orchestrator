# Task W97-N: Final Hardening For Adapter Probability And Restore Validation

## Context
The final system reviewer found no release blockers, but left three senior-quality residual findings:
- P2: optional `SklearnTextClassifierAdapter` still normalizes count-like `predict_proba` rows;
- P3: resume bridge catches any `TypeError` from `restore_round_samples` and can hide internal TypeErrors;
- P3: WAIT path relies on backend-specific task-id validation before poll.

## Goal
Close these residual findings so the final acceptance report has no known real code-quality complaints in the reviewed scope.

## Ownership
May change:
- `src/active_learning_sdk/adapters/sklearn.py`
- `src/active_learning_sdk/engine.py`
- focused tests under `tests/`

Must not change:
- benchmark harnesses, except if a test import break forces a tiny compatibility update;
- README/docs, unless final test counts need updating later by the orchestrator;
- unrelated strategies/backends.

## Scope
In scope:
- Make `SklearnTextClassifierAdapter._validate_probability_rows(...)` strict: finite, non-negative, 2D, expected row count, class count, and sum-to-1.0 within SDK tolerance; do not renormalize arbitrary positive rows.
- Preserve valid sklearn outputs and existing decision-function probability conversion.
- Replace broad `TypeError` fallback around `restore_round_samples(...)` with signature/capability-based dispatch that preserves legacy two-argument backends without swallowing internal `TypeError`s from three-argument implementations.
- Add generic task-id mapping validation before WAIT/PULL backend calls where feasible: selected ids and task_ids must match exactly for active pushed/polled rounds, with no missing/extra ids.
- Add focused tests for adapter strict rejection, internal TypeError propagation, legacy restore compatibility, and WAIT pre-poll corrupted mapping rejection.

Out of scope:
- Changing external Label Studio task-id format.
- Removing legacy restore compatibility entirely.
- Large runtime refactors unrelated to this hardening.

## Acceptance Criteria
- New focused tests fail on the old behavior and pass after the fix.
- Existing adapter/backend/runtime tests pass.
- Full suite passes.
- No known P2/P3 findings remain in this scope.
