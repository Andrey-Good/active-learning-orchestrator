# R93 - Review W73 Managed Docker And Timeout Enforcement

## Context

W73 added:

- package inclusion for managed Label Studio compose/nginx assets;
- offline tests for `ManagedLabelStudioRuntime`;
- engine WAIT timeout enforcement using `AnnotationPolicy.timeout_seconds` and `on_timeout`.

## Goal

Review W73 for correctness, resumability, package safety, and timeout semantics before Stage 7 is considered complete.

## Responsibility Boundaries

You are a reviewer. Do not edit files.

## In Scope

- `pyproject.toml`
- `src/active_learning_sdk/configs.py`
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/backends/managed_docker.py`
- `tests/test_managed_label_studio.py`
- `tests/test_annotation_timeouts.py`
- managed Label Studio asset paths if referenced by packaging/tests.

## Required Review Questions

- Are managed Label Studio assets actually included in wheel/sdist and loaded through package resources?
- Do managed runtime tests avoid requiring real Docker while still checking command/env/path behavior?
- Is timeout enforcement resumable across process restarts?
- Does `on_timeout='raise'` raise a clear error without corrupting round state?
- Does `on_timeout='needs_review'` avoid creating fake labels and persist useful timeout trace/details?
- Does `on_timeout='accept_latest'` pull current annotations only, leaving unresolved samples in safe statuses?
- Are timeout validations correct for negative timeout values or unsupported modes?
- Does normal non-timeout WAIT behavior remain unchanged?

## Explicitly Forbidden

- Do not edit files.
- Do not broaden into Stage 8 reporting.

## Validation To Run

- `uv run --group dev pytest -q tests/test_managed_label_studio.py tests/test_annotation_timeouts.py`
- `uv run --group dev pytest -q`
- `uv build --sdist --wheel --out-dir <temp>` if feasible.

## Output

Return severity-ordered findings with file/line refs. If no findings remain, say so explicitly and include validation results.
