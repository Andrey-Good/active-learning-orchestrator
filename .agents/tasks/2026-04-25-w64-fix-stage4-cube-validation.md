# W64 - Fix Stage 4 Probability Cube Validation

## Context
R80 found two Stage 4 core issues:
- stochastic/committee probability cubes accept inconsistent pass/member counts across samples;
- non-normalized probability rows are silently normalized, masking adapter bugs.

## Goal
Make stochastic/committee probability cube validation strict and add regression tests.

## Responsibility Boundaries
Own only Stage 4 validation fix and tests.

## In Scope
- `src/active_learning_sdk/strategies/stochastic.py`
- `tests/test_stochastic_committee_strategies.py`

## Out of Scope
- Do not edit adapters/base unless absolutely required.
- Do not edit engine unless absolutely required.
- Do not edit benchmarks.
- Do not edit docs/README/dependencies.

## Required Fixes
- Enforce consistent pass/member count across all samples in a cube.
- For stochastic predictions, enforce the output honors requested `n_passes` where the strategy requested it.
- For committee predictions, enforce consistent committee member count across samples; no fixed count is required unless exposed by config.
- Reject probability rows that do not sum to 1 within existing tolerances.
- Keep rejecting negative, non-finite, non-numeric, bool, empty, ragged, and row-count mismatches.
- Error messages should identify strategy/method and the mismatch.

## Tests
Add tests covering:
- Stochastic output with fewer/more passes than requested raises `ConfigurationError`.
- Stochastic output with inconsistent pass counts across samples raises `ConfigurationError`.
- Committee output with inconsistent member counts across samples raises `ConfigurationError`.
- Probability row sum not close to 1 raises `ConfigurationError`.
- Existing positive formula tests still pass.

## Validation
- `uv run --group dev pytest -q tests/test_stochastic_committee_strategies.py`
- `uv run --group dev pytest -q`

## Forbidden Actions
- Do not run destructive git commands.
- Do not revert unrelated changes.

## Acceptance Criteria
- R80 P2 findings are fixed.
- Full tests pass.
