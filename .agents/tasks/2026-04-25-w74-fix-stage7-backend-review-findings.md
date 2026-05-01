# W74 - Fix R91 Backend Review Findings

## Context

R91 reviewed W72 and found two P3 issues:

- `_LabelStudioHttpClient.request()` wraps `ValueError` for non-strict JSON payloads but lets `TypeError` escape for non-serializable objects.
- `SimulatorLabelBackend.submit_annotation()` stores non-finite manual scores directly.

## Goal

Close both review findings with focused tests.

## Responsibility Boundaries

You may change:

- `src/active_learning_sdk/backends/label_studio.py`
- `src/active_learning_sdk/backends/simulator.py`
- `tests/test_label_backends.py`

Do not edit managed Docker runtime/assets or engine timeout code.

## Acceptance Criteria

- Non-JSON-serializable Label Studio payloads raise `_LabelStudioApiError` before network calls.
- Simulator manual annotation scores `NaN`/`Infinity` are normalized to `None`.
- Backend tests and full suite pass.

## Validation

- `uv run --group dev pytest -q tests/test_label_backends.py`
- `uv run --group dev pytest -q`
