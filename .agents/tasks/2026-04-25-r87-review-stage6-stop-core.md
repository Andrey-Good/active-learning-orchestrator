# R87 - Review Stage 6 Stop Criteria Core

## Context
W68 implemented extended stop criteria and persisted stop traces.

## Goal
Review stop criteria core for correctness, no-test-data safety, trace persistence, validation, and edge cases.

## Responsibility Boundaries
- This is a read-only review.
- Focus on W68-owned files.

## In Scope
- `src/active_learning_sdk/configs.py`
- `src/active_learning_sdk/engine.py`
- `tests/test_stop_criteria.py`

## Out of Scope
- Do not edit files.
- Do not review benchmark wiring; not implemented yet.
- Do not implement Stage 7.

## Review Questions
- Do min gates prevent premature plateau/acquisition/distribution/calibration stops?
- Are max_labeled/max_rounds still correct?
- Does metric/calibration plateau use only metrics_history?
- Does label distribution use completed/resolved rounds, not test data?
- Does acquisition convergence avoid stopping when no score trace exists?
- Does every stop decision persist a useful trace in `scheduler_state["stop_trace"]`?
- Are invalid criteria rejected?
- Do tests cover the requested cases?

## Validation
- `uv run --group dev pytest -q tests/test_stop_criteria.py`
- `uv run --group dev pytest -q`

## Forbidden Actions
- Do not edit files.
- Do not run destructive git commands.

## Acceptance Criteria
- No blocking Stage 6 stop-core findings remain.
