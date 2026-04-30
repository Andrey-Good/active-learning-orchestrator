# Task W112 - Adapters/Config/API/Strategy Fixes

## Context

The 2026-04-28 audit found packaging/API/config failures plus group-metadata strategy correctness failures.

## Goal

Fix public adapter import errors, optional dependency metadata, config validators, HF batch behavior, prelabel width validation, and group metadata fail-closed strategy behavior.

## Ownership

May edit:
- `pyproject.toml`
- `src/active_learning_sdk/adapters/__init__.py`
- `src/active_learning_sdk/adapters/base.py`
- `src/active_learning_sdk/adapters/huggingface.py`
- `src/active_learning_sdk/configs.py`
- `src/active_learning_sdk/engine.py` only for prelabel width validation if needed.
- `src/active_learning_sdk/strategies/uncertainty.py`
- `src/active_learning_sdk/strategies/hybrid.py`
- tests only if necessary, but prefer production fixes.

Must not edit:
- Label Studio/Docker files.
- State/cache/lock files.

## Failing Tests To Fix

- `tests/test_deep_audit_public_api_packaging_2026_04_28.py::test_advertised_xxhash_extra_is_declared_in_package_metadata`
- `tests/test_deep_audit_public_api_packaging_2026_04_28.py::test_missing_sklearn_extra_reports_actionable_public_adapter_error`
- `tests/test_deep_audit_strategy_correctness_2026_04_28.py::test_group_diverse_entropy_rejects_misordered_group_lookup_results`
- `tests/test_deep_audit_strategy_correctness_2026_04_28.py::test_class_group_balanced_entropy_rejects_misordered_group_lookup_results`
- `tests/test_deep_audit_strategy_correctness_2026_04_28.py::test_hybrid_group_balance_rejects_incomplete_or_foreign_group_lookup_results`
- `tests/test_objection_sweep_adapters_config_api_2026_04_28.py::*`

## Requirements

- Add the advertised `xxhash` optional dependency and include it in `all`.
- Lazy public sklearn adapter import must raise an actionable message mentioning `active-learning-sdk[sklearn]` and `SklearnTextClassifierAdapter`.
- Runtime-checkable minimal adapter protocol must match documented MVP required methods only.
- Split random ratios, annotation policy numeric fields, and managed ports must be type/range checked with `ConfigurationError`.
- HF adapter must normalize zero/negative batch sizes consistently and validate output row count.
- Prelabeling must validate probability row width against `LabelSchema.labels` before backend push.
- Group metadata lookups must validate exact returned sample IDs, order, duplicates, missing and foreign IDs before using groups.

## Validation

Run:

```powershell
uv run pytest tests\test_deep_audit_public_api_packaging_2026_04_28.py tests\test_deep_audit_strategy_correctness_2026_04_28.py tests\test_objection_sweep_adapters_config_api_2026_04_28.py -q
```

Also run static/syntax checks for touched modules.

## Notes

You are not alone in the codebase. Coordinate with W111 if touching `engine.py`; limit any engine edit to prelabel probability validation.
