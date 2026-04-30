# W69 - Fix Stage 6 Stop Trace And Acquisition Convergence

## Context
R87 found Stage 6 stop-core issues:
- exhausted-pool `StopCriteriaReached` from `run_step()` does not persist a stopped trace;
- acquisition convergence can use sparse old/incomplete rounds instead of recent completed rounds with scores;
- non-stop traces discard observations from configured checks.

## Goal
Fix stop trace persistence and acquisition convergence semantics.

## Responsibility Boundaries
Own only stop trace fixes and tests.

## In Scope
- `src/active_learning_sdk/engine.py`
- `tests/test_stop_criteria.py`

## Out of Scope
- Do not edit configs unless absolutely required.
- Do not edit benchmarks/docs/adapters/backends/dependencies.

## Required Fixes
- When `run()` catches `StopCriteriaReached`, write and persist a stopped trace with reason from the exception where possible, e.g. `no_unlabeled_samples`.
- Acquisition convergence must:
  - consider only completed rounds (`RoundStatus.DONE`);
  - require the most recent configured number of completed rounds to each have the acquisition score key;
  - not stop if any recent required round is missing the score key.
- Non-stop traces should preserve diagnostic observations from configured checks, e.g. metric plateau values, acquisition score values/reason, label distribution deltas, calibration values.
- Keep trace JSON-serializable.

## Tests
Add tests for:
- exhausted pool run stop writes stopped trace.
- acquisition convergence ignores incomplete/failed rounds.
- acquisition convergence does not stop when recent completed rounds are missing score key.
- non-stop trace includes observations for configured criteria.
- Existing stop tests still pass.

## Validation
- `uv run --group dev pytest -q tests/test_stop_criteria.py`
- `uv run --group dev pytest -q`

## Forbidden Actions
- Do not run destructive git commands.
- Do not revert unrelated changes.

## Acceptance Criteria
- R87 P2/P3 findings are fixed.
- Full tests pass.
