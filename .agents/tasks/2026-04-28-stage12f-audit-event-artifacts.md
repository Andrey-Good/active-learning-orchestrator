# Task: stage12f-audit-event-artifacts

## Context

Stage 12C found that the SDK has strict current-state reports but not a professional post-hoc audit trail. A release-quality active learning SDK must explain state transitions, backend/cache operations, and acquisition choices after the run.

## Goal

Add a bounded, deterministic audit artifact layer: append-only events and per-round selection audit artifacts, then surface them in reports/manifests.

## Ownership

May change:

- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/report.py`
- `src/active_learning_sdk/state/store.py`
- `src/active_learning_sdk/cache.py` only if needed for cache audit hooks
- `src/active_learning_sdk/utils.py` only for small atomic/hash helpers
- focused tests under `tests/`
- docs/README report sections

Do not change:

- strategy scoring algorithms except for additive diagnostics capture
- Label Studio backend implementation except for consuming backend audit data from Stage 12E
- benchmark result logic

## Required Fixes

- Add an append-only `events.jsonl` or equivalent event log in project `workdir` with schema version, monotonic index, timestamp, event type, round id, previous/new status where relevant, and strict-JSON-safe metadata.
- Emit events for configure, attach runtime, import labels, round creation, SELECT/PUSH/WAIT/PULL/TRAIN/UPDATE transitions, stop decisions, cache clears/automatic invalidations, report generation, and backend operations available from Stage 12E.
- Add per-round selection audit artifacts containing eligible pool count/hash, selected ids, unselected count/hash or bounded ids, scheduler snapshot, diagnostics, tie/fallback/refill metadata where available, and artifact hash/path.
- Include event log and selection artifact references/hashes in report summary and manifest. Include state hash and artifact hashes where available.
- Add basic state migration fixture/tests if new state fields are introduced, while preserving v1 compatibility for existing state files.

## Acceptance Criteria

- A completed simulator run can be audited from generated artifacts: state, events, report manifest, and selection audit artifact.
- Event log is append-only in normal use and strict JSON safe.
- Reports remain deterministic enough for tests except timestamp/hash fields with explicit handling.
- Existing reports and state loading remain backward compatible.

## Expected Validation

- Add focused tests for event ordering, selection audit artifact, report manifest hash references, cache invalidation event, and legacy state load.
- `uv run pytest tests/test_report_generation.py tests/test_state_safety.py -q`
- Full suite before final Stage 12 acceptance.

## Dependencies

Start after Stage 12D and Stage 12E because it should record the final semantics and backend audit fields, not intermediate contracts.
