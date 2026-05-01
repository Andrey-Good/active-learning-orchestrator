# Task W106: Fix Current P1/P2 Xfails In Engine/Strategy Contracts

## Context

`docs/SENIOR_SDK_ALL_OBJECTIONS_BACKLOG_2026-04-27.md` identifies five current xfail acceptance defects in `tests/test_senior_acceptance_remaining_defects_2026_04_27.py`.

## Goal

Fix the five current xfail tests in production code. Do not edit docs. Do not remove xfail markers; integration will do that after validation.

## Ownership

You may edit:

- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/strategies/hybrid.py`
- `src/active_learning_sdk/strategies/stochastic.py`
- `src/active_learning_sdk/strategies/uncertainty.py` only if shared probability contract requires it

Do not edit:

- tests
- docs
- README
- backend/provider/config/state files

## Required Fixes

1. `SelectionContext.embed()` no-cache path must validate like cached path and return cleaned `list[float]` rows.
2. Hybrid probability validation must enforce `LabelSchema.labels` width.
3. `SelectionContext.predict_proba()` cached rows must be validated against current label schema width; stale mismatches must be evicted and recomputed.
4. `attach_runtime()` must reject `group_id` drift when group-aware strategy semantics depend on it.
5. `configure()` must reject unsafe reconfiguration after rounds or labels already exist.

## Constraints

- Preserve existing public APIs.
- Prefer explicit `ConfigurationError` or `DatasetMismatchError`.
- Keep cached and uncached validation semantics identical.
- Do not weaken previous train-only acquisition or column-split drift checks.

## Acceptance Criteria

- `uv run pytest tests/test_senior_acceptance_remaining_defects_2026_04_27.py -q --runxfail` should pass.
- Relevant existing tests around hard audit/import labels/cache should still pass.
- Return changed files and validations.
