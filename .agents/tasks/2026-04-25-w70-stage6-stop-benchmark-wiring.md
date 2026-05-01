# W70 - Stage 6 Stop Policy Benchmark Wiring

## Context

Stage 6 added SDK stop criteria and persistent stop traces. The feature still needs benchmark evidence that answers product questions:

- How many labels can auto-stop save compared with using the full label budget?
- How much quality is lost, preserved, or gained at the stop point?
- Which stop-policy settings are too aggressive versus useful?

The existing `benchmarks/sdk_first_benchmark.py` already produces deterministic quality curves across datasets, strategies, budgets, and seeds. W70 should add stop-policy simulation on top of those curves, not create notebooks.

## Goal

Add a deterministic stop-policy benchmark layer to `benchmarks/sdk_first_benchmark.py` and focused tests proving it emits actionable stop-vs-fixed-budget diagnostics.

## Responsibility Boundaries

You own:

- `benchmarks/sdk_first_benchmark.py`
- `tests/test_sdk_first_benchmark_embedding_diagnostics.py`
- `benchmarks/README.md` if needed for artifact description

Do not edit SDK stop-criteria core files unless you discover a hard blocker and report it first.

## In Scope

- Add small dataclasses/functions for benchmark stop policies.
- Simulate stop decisions per `(dataset, strategy, seed)` curve using already generated `metrics_rows`.
- Emit one row per stop policy and curve with:
  - policy name and type;
  - stop reason;
  - stopped budget and full/final budget;
  - labels saved and relative savings;
  - selected metric at stop and at full budget;
  - quality delta vs full budget;
  - runtime at stop, runtime full, and runtime saved if available.
- Write `stop_policies.csv` and include stop-policy rows in `summary.json` / `summary.md` / manifest.
- Add tests for:
  - plateau stop triggers after configured patience and min budget;
  - no stop falls back to final budget with zero savings;
  - generated rows are strict-JSON serializable;
  - tiny benchmark curve can produce stop-policy rows.

## Out Of Scope

- Do not add external dependencies.
- Do not run long full benchmarks as part of tests.
- Do not use test-label metrics during acquisition selection. Stop simulation is post-hoc benchmark analysis of curves.
- Do not add notebooks.
- Do not change existing strategy formulas.

## Architectural Constraints

- Keep the benchmark deterministic for fixed seeds.
- Keep generated artifacts as plain CSV/JSON/Markdown.
- Preserve existing CLI behavior; smoke/full should still run without extra required arguments.
- Use metric names that already exist in `metrics_rows` (`macro_f1`, `accuracy`, `weighted_f1`, `rare_recall`).
- Handle NaN metrics safely.

## High-Level Execution Plan

1. Inspect current metric-row schema and summary writer.
2. Add a `StopPolicySpec` dataclass and default stop policy list.
3. Implement `simulate_stop_policies(metrics_rows, policies)` or equivalent.
4. Wire generated rows into `main(...)`, `write_summary(...)`, and artifact descriptions.
5. Add focused tests.
6. Run targeted tests and full pytest.

## Acceptance Criteria

- `benchmarks/sdk_first_benchmark.py` writes `stop_policies.csv` for normal smoke/full runs.
- Stop rows include label-savings and quality-delta columns.
- Summary Markdown contains a compact stop-policy section.
- Tests cover aggressive, conservative/no-stop, and integration paths.
- Existing benchmark smoke and project smoke tests continue to pass.

## Validation

- `uv run --group dev pytest -q tests/test_sdk_first_benchmark_embedding_diagnostics.py`
- `uv run --group dev pytest -q`
- Optional smoke command with temp output:
  `uv run python benchmarks/sdk_first_benchmark.py --preset smoke --datasets grouped_duplicates --strategies random,entropy --budgets 12,18 --seeds 13 --output-dir <temp>`

## Dependencies

- Depends on W68/W69 stop-criteria core being reviewed cleanly.
- Can run in parallel with R88 read-only review because write scope is benchmark/test docs only.
