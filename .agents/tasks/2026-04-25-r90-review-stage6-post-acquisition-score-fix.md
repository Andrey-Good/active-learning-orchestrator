# R90 - Post-Fix Review Stage 6 Acquisition Score Robustness

## Context

R89 found that non-finite acquisition scores (`NaN`, `Infinity`) could enter stop traces and break strict JSON serialization. The orchestrator fixed `_snapshot_numeric_score()` to reject non-finite values and added a focused test.

## Goal

Verify the R89 P2 finding is closed and no Stage 6 regression was introduced.

## Responsibility Boundaries

You are a reviewer. Do not edit files.

## In Scope

- `src/active_learning_sdk/engine.py`
- `tests/test_stop_criteria.py`
- Stage 6 benchmark wiring only if needed to confirm full validation.

## Required Review Questions

- Are non-finite acquisition scores skipped before entering `scores` and stop traces?
- Does the trace remain strict JSON serializable with `allow_nan=False`?
- Does acquisition convergence still work with finite `score_mean` and fallback `scores` lists?
- Do existing Stage 6 tests still pass?

## Validation To Run

- `uv run --group dev pytest -q tests/test_stop_criteria.py`
- `uv run --group dev pytest -q`

## Output

Return findings ordered by severity. If no findings remain, say so explicitly and include validation results.
