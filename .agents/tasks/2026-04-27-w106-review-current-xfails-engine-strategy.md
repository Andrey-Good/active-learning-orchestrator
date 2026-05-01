# Task W106-Review: Engine/Strategy Current-Xfail Fix Review

## Context

W106 fixed the five current acceptance xfails from `tests/test_senior_acceptance_remaining_defects_2026_04_27.py`.

Current validation:

- `uv run pytest tests/test_senior_acceptance_remaining_defects_2026_04_27.py tests/test_w107_nonengine_contracts.py tests/test_label_backends.py -q` -> `28 passed`
- `uv run pytest -q` -> `397 passed`
- `uv build` -> success

## Goal

Perform a read-only senior review of the engine/strategy fixes.

## Scope

Review:

- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/strategies/hybrid.py`
- `tests/test_senior_acceptance_remaining_defects_2026_04_27.py`

## Focus Areas

- Cached and uncached probability validation should enforce current label schema width and evict stale cache rows.
- Cached and uncached embedding validation should return safe cleaned rows.
- Reconfiguration rejection should not block initial configuration but must block semantic reconfigure after labels/rounds.
- Group-aware strategy runtime attach should reject `group_id` drift without overblocking unrelated strategies.
- Hybrid probability validation should match ordinary uncertainty validation.

## Constraints

- Review only. Do not edit files.
- Provide concrete findings with severity and file/line references, or state no material findings remain.
