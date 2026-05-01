# W73 - Stage 7 Managed Docker Packaging And Timeout Enforcement

## Context

Stage 7 aims to make human-labeling workflows reliable. W72/W74 hardened backend contracts and Label Studio HTTP behavior. Remaining Stage 7 product risks:

- Managed Docker assets may not be packaged into the wheel because `pyproject.toml` does not explicitly include compose/nginx non-Python assets.
- Managed Docker runtime has no direct tests for compose command/env/path behavior.
- `AnnotationPolicy.timeout_seconds` / `on_timeout` exists but the engine WAIT step does not enforce it.

## Goal

Make managed Docker mode package-safe/tested and enforce annotation timeouts in the engine state machine.

## Responsibility Boundaries

You may change:

- `src/active_learning_sdk/backends/managed_docker.py`
- `src/active_learning_sdk/backends/assets/**`
- `docker/label_studio/**` only if needed for asset parity
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/configs.py` only if validation gaps are found
- `pyproject.toml`
- tests under `tests/`, preferably `tests/test_managed_label_studio.py` and/or `tests/test_annotation_timeouts.py`

Do not edit Label Studio HTTP parsing/retry code unless a managed-runtime test requires a tiny integration adjustment.

## In Scope

Managed Docker:

- Add tests proving `ManagedLabelStudioRuntime`:
  - resolves packaged/default compose assets into runtime home;
  - exports expected env vars for host port, username, password, token;
  - builds correct compose project command for default and custom compose file names;
  - reports clear errors when compose assets or Docker Compose are unavailable;
  - does not require a real Docker daemon in tests.
- Ensure `docker-compose.yml` and `nginx.conf` are included in wheel/sdist package data.
- Keep packaged `src/.../assets/label_studio` and root `docker/label_studio` assets in sync or document intentional differences.

Timeout enforcement:

- Enforce `AnnotationPolicy.timeout_seconds` in WAIT step using round `created_at` or a persisted wait-start timestamp.
- Respect `AnnotationPolicy.on_timeout`:
  - `raise`: raise a clear backend/active-learning error.
  - `needs_review`: transition unresolved tracked samples to `NEEDS_REVIEW` and complete the round safely.
  - `accept_latest`: allow pulling current annotations if any are available; samples with no annotations should become `NEEDS_REVIEW` or remain unlabeled with a clear trace, whichever best fits existing aggregator semantics.
- Persist timeout traces/details in `round_state.backend_payload` or another existing state field so reports can later surface timeout behavior.
- Add tests for each timeout mode using a fake backend and fast synthetic timestamps.

## Out Of Scope

- Do not run live Docker/Label Studio integration tests unless the environment already has Docker and it is cheap; prefer offline tests here.
- Do not create README release docs in this subtask.
- Do not add external dependencies.

## Architectural Constraints

- Timeout behavior must be resumable; after a restart, an overdue WAIT round should still time out.
- Avoid creating synthetic labels. Timeout handling should never pretend unknown labels are valid training labels.
- Preserve existing non-timeout WAIT behavior.
- Keep tests deterministic and fast.

## Acceptance Criteria

- Managed Docker runtime behavior has offline tests.
- Package config includes Label Studio runtime assets.
- WAIT step enforces all documented timeout modes.
- Timeout state/details are persisted.
- Targeted tests and full suite pass.

## Validation

- `uv run --group dev pytest -q tests/test_managed_label_studio.py tests/test_annotation_timeouts.py`
- `uv run --group dev pytest -q`

## Dependencies

- W72/W74/R92 backend contract hardening completed cleanly.
