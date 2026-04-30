# R83 - Final Stage 4 System Review

## Context
Stage 4 goal: stochastic uncertainty and committee disagreement strategies.

Completed subtasks:
- W63 implemented Stage 4 core strategies.
- R80 found strict cube validation issues.
- W64 fixed probability cube validation.
- R81 reviewed the fixes with no findings.
- W65 wired Stage 4 strategies into benchmarks.
- R82 reviewed benchmark wiring with no findings.

## Goal
Perform final Stage 4 system review and decide whether Stage 4 can close.

## Responsibility Boundaries
- This is a read-only system review.
- Review Stage 4 as an integrated product slice.

## In Scope
- Capability contracts:
  - `src/active_learning_sdk/adapters/base.py`
  - `src/active_learning_sdk/engine.py`
- Strategies:
  - `src/active_learning_sdk/strategies/stochastic.py`
  - `src/active_learning_sdk/strategies/__init__.py`
- Tests:
  - `tests/test_stochastic_committee_strategies.py`
  - `tests/test_strategy_capabilities.py`
  - `tests/test_sdk_first_benchmark_embedding_diagnostics.py`
- Benchmarks:
  - `benchmarks/sdk_first_benchmark.py`

## Out of Scope
- Do not edit files.
- Do not implement Stage 5.
- Do not regenerate committed benchmark artifacts.

## Review Questions
- Are all required stochastic and committee strategies implemented, registered, exported, and benchmarkable?
- Do missing capabilities fail fast at configure/attach?
- Is 3D probability cube validation strict and safe?
- Are formulas coherent and covered by tests?
- Do benchmark proxies clearly avoid claiming real MC-dropout/ensemble training?
- Do BALD and committee smoke benchmarks pass with runtime/selection diagnostics?
- Does the full suite pass?

## Validation
- `uv run --group dev pytest -q`
- `uv run python benchmarks/sdk_first_benchmark.py --preset smoke --strategies random,bald --datasets grouped_duplicates --budgets 12 --seeds 13 --output-dir <temp path>`
- `uv run python benchmarks/sdk_first_benchmark.py --preset smoke --strategies random,committee_vote_entropy --datasets grouped_duplicates --budgets 12 --seeds 13 --output-dir <temp path>`

## Acceptance Criteria
- No open blockers remain for Stage 4.
- Reviewer explicitly says Stage 4 can close if clean.
