# R81 - Review Stage 4 Core After Cube Validation Fix

## Context
W64 fixed R80's findings by enforcing strict stochastic/committee probability cube validation.

## Goal
Verify R80 findings are closed and Stage 4 core is clean.

## Responsibility Boundaries
- This is a read-only review.
- Focus on Stage 4 core validation and formulas.

## In Scope
- `src/active_learning_sdk/strategies/stochastic.py`
- `tests/test_stochastic_committee_strategies.py`
- `tests/test_strategy_capabilities.py`

## Out of Scope
- Do not edit files.
- Do not review benchmark wiring.

## Required Checks
- Stochastic output must honor requested `n_passes`.
- Stochastic output must reject inconsistent pass counts across samples.
- Committee output must reject inconsistent member counts across samples.
- Non-normalized probability rows must be rejected.
- Positive formula tests still pass.

## Validation
- `uv run --group dev pytest -q tests/test_stochastic_committee_strategies.py tests/test_strategy_capabilities.py`
- `uv run --group dev pytest -q`

## Acceptance Criteria
- No findings remain from R80.
