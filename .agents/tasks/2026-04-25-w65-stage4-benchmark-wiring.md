# W65 - Stage 4 Benchmark Wiring For Stochastic And Committee Strategies

## Context
Stage 4 core stochastic and committee strategies are implemented and reviewed. The benchmark harness must now run them with deterministic adapter-provided predictions and report quality/runtime tradeoffs.

## Goal
Wire stochastic and committee strategies into `benchmarks/sdk_first_benchmark.py` with deterministic cheap prediction providers and tests.

## Responsibility Boundaries
Own benchmark wiring only.

## In Scope
- `benchmarks/sdk_first_benchmark.py`
- New or updated benchmark tests under `tests/`

## Out of Scope
- Do not edit SDK core source.
- Do not edit docs/README.
- Do not edit dependency files.
- Do not regenerate committed benchmark artifacts.

## Required Benchmark Behavior
- `SklearnTextBenchmarkAdapter` must expose:
  - `predict_stochastic(texts, n=10, batch_size=...)`
  - `predict_committee(texts, batch_size=...)`
- These methods must be deterministic and strict probability outputs:
  - each stochastic sample has exactly `n` passes;
  - each committee sample has a fixed member count;
  - rows sum to 1;
  - values are finite and non-negative.
- Code comments must state these are benchmark proxies, not real MC-dropout or independently trained committees.
- `strategy_specs()` must include all Stage 4 strategy names:
  - `mc_dropout_entropy`
  - `bald`
  - `variation_ratio`
  - `prediction_variance`
  - `committee_vote_entropy`
  - `committee_kl_divergence`
  - `committee_pairwise_disagreement`
  - `committee_margin`
- Smoke CLI runs should work for at least:
  - `random,bald`
  - `random,committee_vote_entropy`
- Existing runtime and selection diagnostics should include these rows.

## Tests
Add tests that:
- all eight Stage 4 strategies are present in `strategy_specs()`;
- tiny benchmark curve with one stochastic strategy completes;
- tiny benchmark curve with one committee strategy completes;
- generated rows include runtime and existing selection diagnostics.

## Validation
- Run new/updated tests.
- Run `uv run python benchmarks/sdk_first_benchmark.py --preset smoke --strategies random,bald --datasets grouped_duplicates --budgets 12 --seeds 13 --output-dir <temp path>`.
- Run `uv run python benchmarks/sdk_first_benchmark.py --preset smoke --strategies random,committee_vote_entropy --datasets grouped_duplicates --budgets 12 --seeds 13 --output-dir <temp path>`.
- Run `uv run --group dev pytest -q`.

## Files That Must Not Be Touched
- `src/**`
- `README.md`
- `docs/**`
- `benchmarks/results/**`
- `pyproject.toml`
- `uv.lock`

## Forbidden Actions
- Do not run destructive git commands.
- Do not modify accepted benchmark artifacts.

## Acceptance Criteria
- Stage 4 strategies are benchmarkable by one command.
- Full tests pass.
