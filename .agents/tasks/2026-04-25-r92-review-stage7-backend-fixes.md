# R92 - Review W74 Backend Fixes

## Context

R91 found two P3 issues in W72. W74 fixed them:

- `_LabelStudioHttpClient` now wraps both `TypeError` and `ValueError` from strict JSON payload encoding.
- `SimulatorLabelBackend.submit_annotation()` normalizes non-finite manual scores to `None`.

## Goal

Verify both R91 findings are closed without regressions.

## Responsibility Boundaries

You are a reviewer. Do not edit files.

## In Scope

- `src/active_learning_sdk/backends/label_studio.py`
- `src/active_learning_sdk/backends/simulator.py`
- `tests/test_label_backends.py`

## Validation To Run

- `uv run --group dev pytest -q tests/test_label_backends.py`
- `uv run --group dev pytest -q`

## Output

Return findings ordered by severity. If no findings remain, say so explicitly and include validation results.
