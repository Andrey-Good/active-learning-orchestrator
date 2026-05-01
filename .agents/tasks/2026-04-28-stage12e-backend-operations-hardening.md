# Task: stage12e-backend-operations-hardening

## Context

Stage 12A found operational blockers around Label Studio and managed Docker. The highest-risk defect is ambiguous Label Studio `PUSH` failure: a task can be created externally, but the SDK may mark the round failed before persisting recoverable task ids.

## Goal

Make backend operations more recoverable, categorized, and diagnosable without unsafe HTTP POST retries.

## Ownership

May change:

- `src/active_learning_sdk/backends/label_studio.py`
- `src/active_learning_sdk/backends/managed_docker.py`
- `src/active_learning_sdk/backends/base.py` only for additive public dataclasses/helpers
- `src/active_learning_sdk/engine.py` only for backend push/poll/pull audit fields and push recovery
- `src/active_learning_sdk/state/store.py` only for additive round audit fields if Stage 12D did not already add them
- `docker/label_studio/docker-compose.yml`
- `src/active_learning_sdk/backends/assets/label_studio/docker-compose.yml`
- focused tests under `tests/`
- docs explaining opt-in live tests

Do not change:

- annotation aggregation semantics already handled by Stage 12D
- strategy algorithms
- benchmark scripts

## Required Fixes

- Keep raw HTTP POST non-retried, but make Label Studio `push_round()` attempt metadata reconciliation after ambiguous create/import failures before surfacing failure.
- Ensure engine does not abandon a recoverable same-round push before trying an idempotent reconciliation path for the same `round_id`.
- Persist bounded backend audit data for normal push/poll/pull paths: backend ref, last poll progress summary, pull payload summary, and backend error summaries without secrets.
- Validate malformed Label Studio API success payloads and raise `LabelBackendError`/`ConfigurationError` with endpoint/context instead of raw `KeyError`/`TypeError`.
- Improve managed Docker readiness diagnostics: compose `ps`, bounded logs, compose path/url/runtime dir, redacted secrets.
- Add opt-in live Label Studio/managed Docker integration test target gated by env vars and skipped by default.

## Acceptance Criteria

- Unit tests cover ambiguous create/import recovery, malformed API responses, persisted backend audit summaries, and managed diagnostics.
- No secrets are written to state, reports, exceptions, or logs.
- Docker compose assets remain in sync.
- Existing backend tests remain green.

## Expected Validation

- `uv run pytest tests/test_label_backends.py tests/test_managed_label_studio.py -q`
- Run new focused backend tests.
- `uv run mypy src`

## Dependencies

Start after Stage 12D is implemented or explicitly coordinate if both need to touch `engine.py`/`state/store.py`.
