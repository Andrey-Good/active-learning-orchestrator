# W68 - Stage 6 Stop Criteria And Decision Trace Core

## Context
Stage 6 adds budget optimization and safe stopping logic. Current stop logic supports only max labeled, max rounds, and a simple metric plateau. It does not persist a trace explaining why the loop stopped and does not support acquisition-score convergence or label-distribution stabilization.

## Goal
Implement robust stop criteria with persisted decision traces while ensuring stop decisions never use test data.

## Responsibility Boundaries
Own the core stop criteria and trace implementation.

## In Scope
- `src/active_learning_sdk/configs.py`
- `src/active_learning_sdk/engine.py`
- New tests in `tests/test_stop_criteria.py`

## Out of Scope
- Do not edit benchmarks yet.
- Do not edit docs/README yet.
- Do not edit state schema dataclasses unless necessary; prefer using existing `scheduler_state` for trace persistence.
- Do not edit adapters/backends.
- Do not edit dependency files.

## Required StopCriteria Fields
Extend `StopCriteria` with:
- `min_labeled: Optional[int] = None`
- `min_rounds: Optional[int] = None`
- `acquisition_score_key: str = "score_mean"`
- `acquisition_score_rounds: Optional[int] = None`
- `acquisition_score_min_delta: float = 0.0`
- `label_distribution_rounds: Optional[int] = None`
- `label_distribution_max_delta: Optional[float] = None`
- `calibration_metric_name: str = "ece"`
- `calibration_rounds: Optional[int] = None`
- `calibration_min_delta: float = 0.0`

Keep existing fields backward-compatible.

## Required Behavior
- `max_labeled` and `max_rounds` still work.
- `min_labeled` and `min_rounds` prevent early stop from plateau/acquisition/distribution/calibration checks until minimums are met.
- Metric plateau:
  - use validation metrics from `metrics_history` only;
  - do not inspect test split or test labels;
  - persist a trace when it stops.
- Acquisition-score convergence:
  - read recent selected-round scheduler snapshots or stop trace data;
  - support a numeric score key such as `score_mean`;
  - stop when recent acquisition scores change by less than threshold for configured rounds.
  - If no acquisition score trace exists, do not stop.
- Label-distribution stabilization:
  - use resolved labels from completed rounds only;
  - compare recent per-round label distributions;
  - stop when max L1 delta is <= configured threshold.
  - If insufficient labeled rounds exist, do not stop.
- Calibration stabilization:
  - use `metrics_history` metric named by `calibration_metric_name`;
  - same no-test-data rule as metric plateau.
- Every stop decision should write a trace into `state.scheduler_state["stop_trace"]` with:
  - timestamp;
  - stopped bool;
  - reason;
  - criteria values;
  - relevant observed values;
  - labeled count;
  - completed round count.
- `run()` should persist the trace before breaking.
- `run_step()` does not need to stop by criteria unless criteria are passed to `run()`; existing behavior can remain.

## Acquisition Score Trace
- During `_step_select`, if a scheduler snapshot includes score fields (for example future strategies), preserve them.
- If no score fields exist, no acquisition convergence stop should trigger.
- It is acceptable for Stage 6 to define expected keys and test with synthetic round snapshots.

## Validation
- `StopCriteria.validate()` must reject invalid combinations:
  - negative min/max values;
  - min greater than max when both provided;
  - rounds fields < 1;
  - negative deltas where invalid.

## Tests
Add tests covering:
- `min_labeled` prevents plateau stop before enough labels.
- `min_rounds` prevents plateau stop before enough rounds.
- metric plateau stop writes trace.
- acquisition-score convergence stop with synthetic scheduler snapshots writes trace.
- no acquisition-score data means no stop.
- label-distribution stabilization stop and insufficient-data no-stop.
- calibration stabilization stop.
- max_labeled still clips effective batch size.
- stop criteria validation errors.
- trace persists in project state after `run()` stops.

## Validation Commands
- `uv run --group dev pytest -q tests/test_stop_criteria.py`
- `uv run --group dev pytest -q`

## Forbidden Actions
- Do not run destructive git commands.
- Do not modify benchmark result artifacts.
- Do not revert unrelated dirty worktree changes.

## Acceptance Criteria
- Full tests pass.
- Stop decisions are traceable and persisted.
- Stop logic uses validation metrics/state only, never test data.
