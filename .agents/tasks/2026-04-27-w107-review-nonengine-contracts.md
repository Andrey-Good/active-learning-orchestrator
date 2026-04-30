# Task W107-Review: Non-Engine Correctness Contract Review

## Context

W107 fixed non-engine correctness contracts around annotation aggregation, label schema validation, DataFrame JSON safety, and Label Studio project compatibility.

Current validation:

- `uv run pytest tests/test_w107_nonengine_contracts.py tests/test_label_backends.py -q` -> passes
- `uv run pytest -q` -> `397 passed`
- `uv build` -> success

## Goal

Perform a read-only senior review of non-engine fixes.

## Scope

Review:

- `src/active_learning_sdk/annotation.py`
- `src/active_learning_sdk/configs.py`
- `src/active_learning_sdk/dataset/provider.py`
- `src/active_learning_sdk/backends/label_studio.py`
- `tests/test_w107_nonengine_contracts.py`
- affected `tests/test_label_backends.py`

## Focus Areas

- `allow_single_annotator=False` should require distinct annotators.
- Majority ties should route to review.
- Label schema validation should reject bad labels with `ConfigurationError`.
- DataFrame provider normalization should be JSON-safe without destroying useful values.
- Label Studio project reuse should fail on incompatible label config rather than silently mutate.

## Constraints

- Review only. Do not edit files.
- Provide concrete findings with severity and file/line references, or state no material findings remain.
