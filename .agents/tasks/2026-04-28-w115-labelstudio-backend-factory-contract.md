# Task W115: Label Studio response/prelabel contracts and custom backend factory error

## Context

The current open-defect sweep added strict xfail tests for:

- Label Studio bare-list project/task pagination responses;
- mapping prelabels accepting bool and negative values;
- `LabelBackendConfig(backend="custom")` validating while `build_label_backend()` reports only unsupported backend instead of the injection contract.

## Goal

Make the Label Studio/backend-factory xfail tests pass under `--runxfail` while preserving existing Label Studio behavior.

## Ownership

You may edit:

- `src/active_learning_sdk/backends/label_studio.py`
- `src/active_learning_sdk/backends/base.py`
- relevant tests only if adding regression coverage is necessary

Do not edit runtime cache/state/fingerprint/split logic, benchmark scripts, docs, or adapters.

## Constraints

- Centralize or make consistent dict/list response normalization.
- Mapping prelabels should reject bool, non-finite, and negative values. They do not have to sum to 1 unless you can do it without breaking intended score-map semantics.
- `backend="custom"` should produce an actionable error that mentions injecting/providing `label_backend`.

## Forbidden Actions

- Do not weaken direct probability-row validation added earlier.
- Do not silently ignore malformed prelabels that should be rejected.
- Do not add network dependencies to unit tests.

## Suggested Validation

- `uv run pytest tests\test_current_open_audit_defects_2026_04_28.py tests\test_label_backends.py tests\test_objection_sweep_security_infra_2026_04_28.py -q --runxfail`
- `uv run pytest tests\test_label_backends.py -q`

## Acceptance Criteria

- The three backend/Label Studio tests pass under `--runxfail`.
- Existing Label Studio idempotency, parsing, and prelabel tests remain green.
