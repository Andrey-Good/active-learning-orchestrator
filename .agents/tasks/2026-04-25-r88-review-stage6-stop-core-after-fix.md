# R88 - Review Stage 6 Stop Core After Fix

## Context
W69 fixed R87 findings around stop trace persistence and acquisition convergence semantics.

## Goal
Verify Stage 6 stop-core findings are closed.

## Responsibility Boundaries
- This is a read-only review.
- Focus on W69 fixes and Stage 6 stop-core semantics.

## In Scope
- `src/active_learning_sdk/engine.py`
- `tests/test_stop_criteria.py`

## Out of Scope
- Do not edit files.
- Do not review benchmark wiring.

## Required Checks
- Exhausted-pool stop writes and persists stopped trace.
- Acquisition convergence uses only recent completed rounds and requires score key in each required recent round.
- Non-stop traces preserve observations.
- Full stop tests pass.

## Validation
- `uv run --group dev pytest -q tests/test_stop_criteria.py`
- `uv run --group dev pytest -q`

## Acceptance Criteria
- No R87 findings remain.
