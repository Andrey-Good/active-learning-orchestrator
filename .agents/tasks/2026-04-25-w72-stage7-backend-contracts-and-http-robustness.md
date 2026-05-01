# W72 - Stage 7 Backend Contracts And HTTP Robustness

## Context

Stage 7 hardens labeling backends for product use. Current backend code exists for Label Studio, managed Docker, and simulator, but there is little direct backend contract coverage. We need to make the backend layer reliable before touching managed Docker packaging/lifecycle.

## Goal

Add backend contract tests and harden the Label Studio/simulator behavior around idempotency, parsing, retryable HTTP failures, strict-JSON-safe values, and deterministic simulator/oracle use.

## Responsibility Boundaries

You may change:

- `src/active_learning_sdk/backends/base.py`
- `src/active_learning_sdk/backends/label_studio.py`
- `src/active_learning_sdk/backends/simulator.py`
- `src/active_learning_sdk/backends/__init__.py`
- `tests/test_label_backends.py` or a similarly named new backend test file

Do not edit managed Docker runtime/assets in this subtask. That is W73.

## In Scope

- Add tests for `SimulatorLabelBackend` lifecycle:
  - `ensure_ready()` validates schema and returns backend metadata.
  - `push_round()` is idempotent for the same `(round_id, sample_id)`.
  - `poll_round()` respects `AnnotationPolicy.min_votes`.
  - `pull_round()` returns sorted annotations.
  - misuse before `ensure_ready()` raises.
- Extend simulator for useful SDK tests if needed:
  - optional `label_by_sample_id` or `label_fn` oracle mode that can auto-populate annotations on `push_round()` or `pull_round()`;
  - preserve manual `submit_annotation()` behavior.
- Add Label Studio unit tests with a fake HTTP client / monkeypatch, without requiring a live Label Studio:
  - project creation/reuse/config patch behavior;
  - task idempotency via metadata;
  - annotation parsing for single-label and multi-label results;
  - prediction import creation and duplicate prediction avoidance;
  - non-finite prediction/annotation scores do not enter persisted payloads.
- Harden `_LabelStudioHttpClient` with bounded retry/backoff for transient connection errors, 408, 429, and 5xx responses.
  - Keep permanent 4xx failures non-retryable.
  - Preserve clear errors with method, URL, status, and response body where available.
  - Make retry/backoff parameters testable without sleeping long.

## Out Of Scope

- Do not start real Docker/Label Studio.
- Do not edit Docker assets or packaging config.
- Do not add external dependencies.
- Do not change engine state-machine behavior unless a backend contract bug forces it.

## Architectural Constraints

- Backend methods must remain deterministic and idempotent where possible.
- Tests must be fast and offline.
- Avoid monkeypatching global network state broadly; prefer fake client objects for `LabelStudioBackend`.
- Do not let `NaN`/`Infinity` appear in backend payloads intended for JSON/state artifacts.
- Keep public API changes small and backwards compatible.

## Acceptance Criteria

- New backend tests cover simulator and Label Studio parsing/idempotency behavior.
- Transient HTTP retries are covered by tests.
- Non-finite scores are skipped or normalized safely.
- Full test suite passes.

## Validation

- `uv run --group dev pytest -q tests/test_label_backends.py`
- `uv run --group dev pytest -q`

## Dependencies

- Starts after Stage 6 completion.
- W73 managed Docker hardening should wait until this is reviewed.
