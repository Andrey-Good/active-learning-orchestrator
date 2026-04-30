# R74 - Review Stage 2 Benchmark Embedding Diagnostics

## Context
W59 updated the SDK-first benchmark harness to include embedding strategies and redundancy diagnostics.

## Goal
Review benchmark changes for correctness, artifact safety, and usefulness for comparing diversity methods.

## Responsibility Boundaries
- This is a read-only review.
- Focus on W59-owned benchmark/test changes.

## In Scope
- `benchmarks/sdk_first_benchmark.py`
- `tests/test_sdk_first_benchmark_embedding_diagnostics.py`
- Temporary-output benchmark CLI validation.

## Out of Scope
- Do not edit files.
- Do not regenerate committed benchmark artifacts.
- Do not review SDK core embedding algorithm internals except if benchmark usage is wrong.

## Review Questions
- Does the benchmark adapter expose deterministic meaningful embeddings?
- Are all five embedding strategies included in `strategy_specs()`?
- Are redundancy metrics strict-JSON safe and present in selection artifacts?
- Do group concentration metrics remain unchanged?
- Does a tiny random-vs-coreset smoke command pass to a temp dir?
- Are tests sufficient and not touching `benchmarks/results/**`?

## Validation
- `uv run --group dev pytest -q tests/test_sdk_first_benchmark_embedding_diagnostics.py`
- `uv run python benchmarks/sdk_first_benchmark.py --preset smoke --strategies random,coreset_kcenter --datasets grouped_duplicates --budgets 12 --seeds 13 --output-dir <temp path>`
- `uv run --group dev pytest -q`

## Forbidden Actions
- Do not edit files.
- Do not run destructive git commands.

## Acceptance Criteria
- No findings remain for Stage 2 benchmark diagnostics.
