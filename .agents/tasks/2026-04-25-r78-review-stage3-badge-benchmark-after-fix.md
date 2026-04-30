# R78 - Review Stage 3 BADGE Benchmark After Runtime Fix

## Context
W62 fixed R77's finding by adding runtime to benchmark selection rows.

## Goal
Verify BADGE benchmark wiring is clean after the runtime fix.

## Responsibility Boundaries
- This is a read-only review.
- Focus on R77 fix and BADGE benchmark evidence.

## In Scope
- `benchmarks/sdk_first_benchmark.py`
- `tests/test_sdk_first_benchmark_embedding_diagnostics.py`
- Temp benchmark smoke output.

## Out of Scope
- Do not edit files.
- Do not regenerate committed benchmark artifacts.

## Validation
- `uv run --group dev pytest -q tests/test_sdk_first_benchmark_embedding_diagnostics.py`
- `uv run python benchmarks/sdk_first_benchmark.py --preset smoke --strategies random,badge --datasets grouped_duplicates --budgets 12 --seeds 13 --output-dir <temp path>`
- Confirm generated `selections.csv` includes runtime and redundancy diagnostics.
- `uv run --group dev pytest -q`

## Acceptance Criteria
- No findings remain for BADGE benchmark wiring.
