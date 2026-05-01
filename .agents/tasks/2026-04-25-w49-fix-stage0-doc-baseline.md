# W49 - Fix Stage 0 Documentation Baseline

## Context
Stage 0 must leave the existing SDK core in a stable, accurately documented state before new capability work begins. Readiness audit R62 found stale documentation values after benchmark and test harness changes.

## Goal
Update documentation so it reflects the current Stage 0 baseline exactly and does not imply missing artifact directories are required accepted outputs.

## Responsibility Boundaries
- This is a documentation-only task.
- Keep edits narrow and factual.

## In Scope
- `README.md`
- `benchmarks/README.md`
- `benchmarks/results/current_benchmark_report.md`

## Out of Scope
- Do not edit SDK source code.
- Do not edit tests.
- Do not edit benchmark scripts.
- Do not edit dependency files.
- Do not regenerate benchmark artifacts.

## Required Changes
- Replace stale accepted test count `42 passed` with the current accepted result `47 passed`.
- Add `tests/test_reference_strategy_benchmark.py` to the README test inventory.
- Clarify benchmark output paths so examples are not confused with accepted committed artifact directories.
- Ensure documented accepted artifact directories match currently available Stage 0 evidence, especially:
  - `benchmarks/results/reference_full`
  - `benchmarks/results/class_group_balanced_entropy`
  - `benchmarks/results/mix_interleaved_probe`
  - `benchmarks/results/project_smoke`
  - `benchmarks/results/smoke`
- If mentioning `benchmarks/results/full` or `benchmarks/results/reference_smoke`, make clear they are example output paths that may be generated locally, not required accepted committed baselines.

## Files That May Be Changed
- `README.md`
- `benchmarks/README.md`
- `benchmarks/results/current_benchmark_report.md`

## Files That Must Not Be Touched
- `src/**`
- `tests/**`
- `benchmarks/*.py`
- `pyproject.toml`
- `uv.lock`
- Docker and Label Studio files
- Existing benchmark result data files except `benchmarks/results/current_benchmark_report.md`

## Important Constraints
- Do not reintroduce notebook workflows as active entrypoints.
- Preserve the user-facing Russian/English style already present in files; do not do a broad rewrite.
- Keep benchmark claims backed by existing artifacts or clearly labeled as commands/examples.

## Forbidden Actions
- Do not run destructive git commands.
- Do not alter benchmark results to fit documentation.
- Do not invent metrics not present in existing artifacts.

## Execution Plan
1. Inspect the three in-scope docs around stale lines.
2. Patch only the stale or ambiguous documentation.
3. Validate by searching for:
   - `42 passed`
   - `test_reference_strategy_benchmark`
   - `benchmarks/results/full`
   - `benchmarks/results/reference_smoke`
   - active notebook references such as `active_learning_lab`.

## Acceptance Criteria
- No stale `42 passed` claim remains.
- README lists all current test files including `tests/test_reference_strategy_benchmark.py`.
- Benchmark documentation distinguishes example output paths from accepted Stage 0 artifact paths.
- No active notebook benchmark entrypoint is documented.

## Dependencies
- Based on R62 readiness audit findings.
- Can run in parallel with R63 review because write scope does not overlap.
