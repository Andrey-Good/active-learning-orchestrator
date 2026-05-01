# W62 - Add Runtime To Benchmark Selection Rows

## Context
R77 found that BADGE benchmark selection rows contain redundancy diagnostics but no runtime column. Runtime is present in `metrics.csv`, but `selections.csv` should also independently expose acquisition/runtime context for strategy diagnostics.

## Goal
Add runtime to selection rows and test that BADGE selection artifacts include it.

## Responsibility Boundaries
Own only benchmark/test fix.

## In Scope
- `benchmarks/sdk_first_benchmark.py`
- `tests/test_sdk_first_benchmark_embedding_diagnostics.py`

## Out of Scope
- Do not edit SDK core.
- Do not edit docs/README.
- Do not regenerate committed benchmark artifacts.

## Required Change
- Add `runtime_seconds` or a clearly named acquisition/runtime column to `selection_rows`.
- Ensure strict JSON/CSV safety.
- Update tests so BADGE selection diagnostics assert runtime presence.

## Validation
- `uv run --group dev pytest -q tests/test_sdk_first_benchmark_embedding_diagnostics.py`
- `uv run python benchmarks/sdk_first_benchmark.py --preset smoke --strategies random,badge --datasets grouped_duplicates --budgets 12 --seeds 13 --output-dir <temp path>`
- `uv run --group dev pytest -q`

## Forbidden Actions
- Do not run destructive git commands.
- Do not modify accepted result artifacts.

## Acceptance Criteria
- R77 P2 is fixed.
- Full tests pass.
