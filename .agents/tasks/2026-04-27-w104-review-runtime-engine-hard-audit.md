# Task W104-Review: Runtime/Engine Hard-Audit Fix Review

## Context

Runtime/engine hard-audit blockers from `docs/SENIOR_SDK_HARD_AUDIT_2026-04-27.md` were fixed in `src/active_learning_sdk/engine.py` and related tests were converted from xfail to normal regressions.

Current validation:

- `uv run pytest tests/test_hard_audit_known_defects_2026_04_27.py tests/test_import_labels.py -q` -> `21 passed`
- `uv run pytest -q` -> `381 passed`
- `uv build` -> success

## Goal

Review the runtime/engine changes for correctness, compatibility, and maintainability.

## Scope

Review:

- `src/active_learning_sdk/engine.py`
- `tests/test_hard_audit_known_defects_2026_04_27.py`
- `tests/test_import_labels.py`
- `tests/test_audit_runtime_edge_cases.py` only where affected by train-only selection

## Focus Areas

- Selection must be train-only by default without breaking existing lifecycle semantics.
- `attach_runtime()` must detect column split drift.
- Cache-disabled `SelectionContext.predict_proba()` must validate and return JSON-safe cleaned rows.
- `status()["active_round"]` must report only real active rounds.
- Seed train idempotency must not skip retraining for volatile/unversioned fresh model instances.

## Constraints

- Review only. Do not edit files.
- Provide concrete findings with severity and file/line references, or state that no material findings remain.
