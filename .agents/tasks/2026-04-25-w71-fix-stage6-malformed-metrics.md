# W71 - Fix Malformed Stop Metric Handling

## Context

R88 found that calibration stop evaluation can crash when `metrics_history` contains malformed values for the configured metric. This breaks Stage 6 robustness expectations.

## Goal

Make metric extraction for stop criteria robust to malformed metric values.

## Responsibility Boundaries

You may change:

- `src/active_learning_sdk/engine.py`
- `tests/test_stop_criteria.py`

Do not edit benchmark files, docs, or unrelated SDK areas.

## In Scope

- Treat missing, `None`, non-numeric, NaN, or infinite metric values as unusable for stop-series calculations.
- Preserve a diagnostic indication that values were skipped if practical.
- Ensure `_should_stop(StopCriteria(calibration_rounds=...))` does not raise on malformed values.
- Add focused tests for malformed calibration metrics and, if appropriate, plateau metrics.

## Out Of Scope

- New stop criteria features.
- Benchmark wiring.
- README changes.

## Acceptance Criteria

- Malformed calibration metric values do not crash `_should_stop`.
- The stop trace remains `criteria_not_met` when insufficient valid values remain.
- Existing stop criteria tests pass.
- Full test suite passes.

## Validation

- `uv run --group dev pytest -q tests/test_stop_criteria.py`
- `uv run --group dev pytest -q`
