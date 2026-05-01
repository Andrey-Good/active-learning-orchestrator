# W75 - Fix R93 Timeout Review Findings

## Context

R93 reviewed W73 and found:

- P2: after `on_timeout='raise'`, if a later retry succeeds, `round_state.error` remains stale.
- P3: `_step_wait` writes `annotation_timeout.wait_started_at` even when timeout enforcement is disabled.

## Goal

Fix both issues and add regression tests.

## Responsibility Boundaries

You may change:

- `src/active_learning_sdk/engine.py`
- `tests/test_annotation_timeouts.py`

Do not edit managed Docker files or backend HTTP code.

## Acceptance Criteria

- Normal WAIT with `timeout_seconds=None` does not add an `annotation_timeout` trace.
- A round that previously timed out with `raise` clears `round_state.error` when a later poll succeeds.
- Targeted timeout tests and full suite pass.

## Validation

- `uv run --group dev pytest -q tests/test_annotation_timeouts.py`
- `uv run --group dev pytest -q`
