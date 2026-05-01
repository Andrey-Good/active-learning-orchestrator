# Task: stage12d-annotation-review-metadata

## Context

Stage 12 audits found that annotation policy semantics are still not release-quality for multi-annotator workflows. `allow_single_annotator=False` can accept a single annotator when `min_votes=1`, simulator polling does not mirror distinct-annotator readiness, and normal pull/timeout paths discard useful `AnnotationAggregator` review details.

## Goal

Make annotation resolution auditable and policy-honest without weakening existing runtime safety.

## Ownership

May change:

- `src/active_learning_sdk/annotation.py`
- `src/active_learning_sdk/configs.py`
- `src/active_learning_sdk/backends/simulator.py`
- `src/active_learning_sdk/state/store.py`
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/report.py`
- focused tests under `tests/`
- small docs updates if needed

Do not change:

- strategy algorithms
- benchmark scripts
- Label Studio HTTP/managed Docker implementation except where tests require simulator parity
- packaging metadata unless required by tests

## Required Fixes

- Make `allow_single_annotator=False` independently prevent a single annotator from becoming an accepted label. Prefer a clear config validation error for `min_votes < 2` unless a better compatible contract is justified.
- Align simulator `poll_round()` readiness with Label Studio: when single annotators are disallowed, readiness must count distinct annotators, not raw annotation count.
- Persist per-sample review metadata for non-labeled annotation resolutions, including reason, agreement, counts/details, annotation count, eligible vote count when available, and policy snapshot.
- Clear stale review metadata when a sample becomes labeled/imported again.
- Include review metadata in `status()`/report summary enough for operational debugging.
- Keep state JSON strict and backward compatible for existing v1 states that lack the new field.

## Acceptance Criteria

- New tests fail on the audited defects before the fix and pass after.
- Existing focused runtime/annotation tests pass.
- Full suite remains green.
- No raw non-JSON-safe values can be saved in state or reports.
- The implementation is deterministic and does not hide invalid backend payloads.

## Expected Validation

- `uv run pytest tests/test_w107_nonengine_contracts.py tests/test_w97_runtime_state_backends.py tests/test_report_generation.py -q`
- Add and run any new focused tests.
- `uv run mypy src`

## Dependencies

This task must complete before backend/event-log work that depends on review reasons.
