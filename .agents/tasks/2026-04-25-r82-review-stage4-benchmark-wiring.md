# R82 - Review Stage 4 Benchmark Wiring

## Context
W65 wired stochastic and committee strategies into the SDK-first benchmark harness.

## Goal
Review benchmark wiring for deterministic proxy correctness, strategy coverage, artifact safety, and validation.

## Responsibility Boundaries
- This is a read-only review.
- Focus on W65 benchmark/test changes.

## In Scope
- `benchmarks/sdk_first_benchmark.py`
- `tests/test_sdk_first_benchmark_embedding_diagnostics.py`
- Temp CLI smoke outputs.

## Out of Scope
- Do not edit files.
- Do not review SDK core implementation; R81 already did.
- Do not regenerate committed benchmark artifacts.

## Review Questions
- Do benchmark proxies return strict probability cubes with fixed pass/member counts?
- Are proxy comments clear that they are not real MC-dropout/ensemble training?
- Are all eight Stage 4 strategy names registered in `strategy_specs()`?
- Do random-vs-BALD and random-vs-committee CLI smokes pass to temp dirs?
- Do generated rows include runtime and selection diagnostics?

## Validation
- `uv run --group dev pytest -q tests/test_sdk_first_benchmark_embedding_diagnostics.py`
- `uv run python benchmarks/sdk_first_benchmark.py --preset smoke --strategies random,bald --datasets grouped_duplicates --budgets 12 --seeds 13 --output-dir <temp path>`
- `uv run python benchmarks/sdk_first_benchmark.py --preset smoke --strategies random,committee_vote_entropy --datasets grouped_duplicates --budgets 12 --seeds 13 --output-dir <temp path>`
- `uv run --group dev pytest -q`

## Acceptance Criteria
- No Stage 4 benchmark wiring findings remain.
