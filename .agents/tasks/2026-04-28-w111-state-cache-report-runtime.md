# Task W111 - State/Cache/Report/Runtime Fixes

## Context

The 2026-04-28 sweep found state validation, export, cache JSON, lock ownership, and backend progress correctness failures.

## Goal

Fix runtime/state/cache/report failures with clear fail-closed behavior.

## Ownership

May edit:
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/cache.py`
- `src/active_learning_sdk/state/lock.py`
- `src/active_learning_sdk/state/store.py`
- tests only if necessary, but prefer production fixes.

Must not edit:
- `src/active_learning_sdk/backends/label_studio.py`
- `src/active_learning_sdk/backends/managed_docker.py`
- `src/active_learning_sdk/backends/assets/*`
- `src/active_learning_sdk/configs.py`
- `src/active_learning_sdk/adapters/*`
- `src/active_learning_sdk/strategies/*`

## Failing Tests To Fix

- `tests/test_deep_audit_runtime_state_backends_2026_04_28.py::test_project_lock_release_from_non_owner_does_not_remove_or_break_active_lock`
- `tests/test_deep_audit_runtime_state_backends_2026_04_28.py::test_wait_rejects_backend_progress_total_that_does_not_match_tracked_tasks`
- `tests/test_objection_sweep_state_cache_report_2026_04_28.py::test_state_store_rejects_boolean_state_version`
- `tests/test_objection_sweep_state_cache_report_2026_04_28.py::test_validate_flags_corrupt_persisted_split_membership`
- `tests/test_objection_sweep_state_cache_report_2026_04_28.py::test_export_dataset_split_rejects_unknown_subset_name`
- `tests/test_objection_sweep_state_cache_report_2026_04_28.py::test_export_labels_refuses_labeled_status_without_label`
- `tests/test_objection_sweep_state_cache_report_2026_04_28.py::test_jsonl_disk_cache_rejects_non_finite_values_before_append`

## Requirements

- `ProjectLock.release()` on a non-owner instance must be a no-op and must not remove another active lock.
- WAIT progress validation must check `total`, `done`, bounds, and ready ids before transitions.
- State version must reject booleans explicitly.
- `validate()` must detect corrupt persisted split duplicates, overlaps, unknown ids, and bad coverage.
- Exports must validate subset names/formats and label invariants before writing.
- JSONL disk cache must use strict JSON and reject non-finite values before append.

## Validation

Run:

```powershell
uv run pytest tests\test_deep_audit_runtime_state_backends_2026_04_28.py tests\test_objection_sweep_state_cache_report_2026_04_28.py -q
```

Also run touched module static/syntax checks.

## Notes

You are not alone in the codebase. Avoid edits outside the declared ownership.
