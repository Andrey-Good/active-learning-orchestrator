# Task W105-Review: Cache/Strategy Hard-Audit Fix Review

## Context

Cache/strategy hard-audit blockers from `docs/SENIOR_SDK_HARD_AUDIT_2026-04-27.md` were fixed in `src/active_learning_sdk/cache.py` and `src/active_learning_sdk/strategies/uncertainty.py`. The known-defect tests now pass as normal regressions.

Current validation:

- `uv run pytest tests/test_hard_audit_known_defects_2026_04_27.py -q` -> passes
- `uv run pytest -q` -> `381 passed`
- `uv build` -> success

## Goal

Review cache/strategy changes for correctness, compatibility, and maintainability.

## Scope

Review:

- `src/active_learning_sdk/cache.py`
- `src/active_learning_sdk/strategies/uncertainty.py`
- `tests/test_hard_audit_known_defects_2026_04_27.py`

## Focus Areas

- `InMemoryCacheStore(max_items=0)` no-op behavior should not hide other cache bugs.
- Probability width validation should correctly use `LabelSchema.labels` where available.
- Existing strict probability validation and deterministic selection must remain intact.
- The change should not accidentally make unrelated strategy classes harder to use.

## Constraints

- Review only. Do not edit files.
- Provide concrete findings with severity and file/line references, or state that no material findings remain.
