# R77 - Review Stage 3 BADGE Benchmark Wiring

## Context
W61 wired BADGE into the SDK-first benchmark harness using a deterministic benchmark-only gradient embedding proxy.

## Goal
Review BADGE benchmark wiring for correctness, artifact safety, and usefulness as Stage 3 evidence.

## Responsibility Boundaries
- This is a read-only review.
- Focus on W61 benchmark/test changes.

## In Scope
- `benchmarks/sdk_first_benchmark.py`
- `tests/test_sdk_first_benchmark_embedding_diagnostics.py`
- Temporary CLI smoke output.

## Out of Scope
- Do not edit files.
- Do not review core BADGE implementation; R76 already did.
- Do not regenerate committed benchmark artifacts.

## Review Questions
- Does `SklearnTextBenchmarkAdapter.gradient_embed` return deterministic contract-valid rows?
- Is the code comment clear that this is a benchmark proxy, not neural autograd?
- Is `badge` included in `strategy_specs()`?
- Does random-vs-BADGE CLI smoke pass to a temp directory?
- Do BADGE selection rows include runtime and redundancy diagnostics?
- Do tests avoid touching `benchmarks/results/**`?

## Validation
- `uv run --group dev pytest -q tests/test_sdk_first_benchmark_embedding_diagnostics.py`
- `uv run python benchmarks/sdk_first_benchmark.py --preset smoke --strategies random,badge --datasets grouped_duplicates --budgets 12 --seeds 13 --output-dir <temp path>`
- `uv run --group dev pytest -q`

## Forbidden Actions
- Do not edit files.
- Do not run destructive git commands.

## Acceptance Criteria
- No findings remain for BADGE benchmark wiring.
