# W61 - Stage 3 BADGE Benchmark Wiring

## Context
BADGE core is implemented and reviewed. Stage 3 exit criteria require BADGE to run in benchmark evidence with a gradient-capable adapter. The SDK contract expects adapters to provide gradient embeddings; the benchmark can use a cheap deterministic proxy to validate SDK behavior and runtime/quality tradeoff.

## Goal
Wire `badge` into the SDK-first benchmark harness with a deterministic `gradient_embed` method and tests.

## Responsibility Boundaries
Own benchmark wiring only.

## In Scope
- `benchmarks/sdk_first_benchmark.py`
- New or updated tests under `tests/` for BADGE benchmark wiring.

## Out of Scope
- Do not edit SDK core source.
- Do not edit README/docs.
- Do not edit dependency files.
- Do not regenerate committed benchmark artifacts.

## Required Benchmark Behavior
- `SklearnTextBenchmarkAdapter` must expose `gradient_embed(texts, labels=None, batch_size=...)`.
- The benchmark gradient embedding should be deterministic, cheap, and contract-valid:
  - using TF-IDF features combined with model uncertainty/pseudo-label information is acceptable;
  - document in code comments that this is a benchmark proxy, not neural autograd.
- `strategy_specs()` must include `badge`.
- A smoke CLI run with `--strategies random,badge` must complete.
- Selection/runtime diagnostics already present must include BADGE rows.

## Tests
Add tests that:
- `strategy_specs()` contains `badge`.
- Tiny benchmark curve with `badge` completes.
- BADGE benchmark selection rows include runtime and redundancy diagnostics.

## Validation
- Run the new/updated tests.
- Run `uv run python benchmarks/sdk_first_benchmark.py --preset smoke --strategies random,badge --datasets grouped_duplicates --budgets 12 --seeds 13 --output-dir <temp path>`.
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
- BADGE is benchmarkable by one command.
- Full tests pass.
