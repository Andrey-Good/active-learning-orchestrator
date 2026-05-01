# R85 - Review Stage 5 Hybrid Benchmark Wiring

## Context
W67 wired representative hybrid strategies into the SDK-first benchmark harness.

## Goal
Review hybrid benchmark wiring for strategy coverage, config correctness, artifact safety, and diagnostics.

## Responsibility Boundaries
- This is a read-only review.
- Focus on W67 benchmark/test changes.

## In Scope
- `benchmarks/sdk_first_benchmark.py`
- `tests/test_sdk_first_benchmark_embedding_diagnostics.py`
- Temp CLI smoke outputs.

## Out of Scope
- Do not edit files.
- Do not review core hybrid implementation; R84 already did.
- Do not regenerate committed benchmark artifacts.

## Review Questions
- Are all four required hybrid benchmark specs present?
- Do specs use `SchedulerConfig(mode="hybrid", hybrid=...)`?
- Do weighted and guarded hybrid CLI smokes pass to temp dirs?
- Do generated rows include quality/runtime/redundancy/group diagnostics?
- Do tests avoid touching `benchmarks/results/**`?

## Validation
- `uv run --group dev pytest -q tests/test_sdk_first_benchmark_embedding_diagnostics.py`
- `uv run python benchmarks/sdk_first_benchmark.py --preset smoke --strategies random,hybrid_weighted_entropy_coreset --datasets grouped_duplicates --budgets 12 --seeds 13 --output-dir <temp path>`
- `uv run python benchmarks/sdk_first_benchmark.py --preset smoke --strategies random,hybrid_weighted_guarded --datasets grouped_duplicates --budgets 12 --seeds 13 --output-dir <temp path>`
- `uv run --group dev pytest -q`

## Acceptance Criteria
- No Stage 5 benchmark wiring findings remain.
