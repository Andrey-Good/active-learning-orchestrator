# W67 - Stage 5 Hybrid Benchmark Wiring

## Context
Stage 5 core hybrid framework is implemented and reviewed. Benchmark evidence must show hybrid modes can run and report quality/guardrail diagnostics against existing baselines.

## Goal
Wire representative hybrid configurations into the SDK-first benchmark harness and add tests.

## Responsibility Boundaries
Own benchmark wiring only.

## In Scope
- `benchmarks/sdk_first_benchmark.py`
- New or updated benchmark tests under `tests/`

## Out of Scope
- Do not edit SDK core.
- Do not edit docs/README.
- Do not edit dependency files.
- Do not regenerate committed benchmark artifacts.

## Required Benchmark Strategies
Add representative strategy specs:
- `hybrid_weighted_entropy_coreset`
- `hybrid_uncertainty_prefilter_coreset`
- `hybrid_diversity_prefilter_entropy`
- `hybrid_weighted_guarded`

Suggested configs:
- weighted entropy + coreset with balanced weights;
- uncertainty prefilter then coreset;
- coreset prefilter then entropy;
- weighted entropy + coreset with class/group guardrails and small exploration fraction.

## Required Behavior
- Strategy specs use `SchedulerConfig(mode="hybrid", hybrid=...)`.
- Existing full preset should include these strategies through `list(available_strategies.keys())`.
- Smoke preset does not need to include all by default.
- Existing quality/runtime/redundancy/group diagnostics should appear for hybrid rows.
- No committed benchmark artifacts should be modified.

## Tests
Add tests that:
- all four hybrid benchmark specs are present;
- a tiny curve with a hybrid weighted strategy completes;
- a tiny curve with a guarded hybrid strategy completes and emits group/redundancy/runtime diagnostics.

## Validation
- Run new/updated tests.
- Run `uv run python benchmarks/sdk_first_benchmark.py --preset smoke --strategies random,hybrid_weighted_entropy_coreset --datasets grouped_duplicates --budgets 12 --seeds 13 --output-dir <temp path>`.
- Run `uv run python benchmarks/sdk_first_benchmark.py --preset smoke --strategies random,hybrid_weighted_guarded --datasets grouped_duplicates --budgets 12 --seeds 13 --output-dir <temp path>`.
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
- Hybrid strategies are benchmarkable by one command.
- Full tests pass.
