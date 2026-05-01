# W50 - Freeze Stage 0 Baseline Evidence

## Context
Stage 0 must end with a concrete, reviewable baseline: current tests pass, benchmark scripts are import/compile-safe, notebook artifacts are removed, and accepted benchmark artifact directories are documented.

## Goal
Create a concise Stage 0 baseline evidence file from existing repository artifacts and validation commands.

## Responsibility Boundaries
- This is a validation and documentation task.
- Do not alter SDK behavior, tests, benchmark scripts, dependencies, or benchmark result data.

## In Scope
- Run Stage 0 validation commands.
- Inspect existing benchmark result artifacts.
- Create `benchmarks/results/stage0_baseline_summary.md`.

## Out of Scope
- Do not run long benchmark experiments unless needed for validation.
- Do not regenerate existing benchmark result CSV/JSON/MD artifacts.
- Do not change README or benchmark README.
- Do not change source code or tests.

## Files That May Be Changed
- `benchmarks/results/stage0_baseline_summary.md`

## Files That Must Not Be Touched
- `src/**`
- `tests/**`
- `benchmarks/*.py`
- `README.md`
- `benchmarks/README.md`
- `benchmarks/results/current_benchmark_report.md`
- Existing benchmark result files under:
  - `benchmarks/results/reference_full/**`
  - `benchmarks/results/class_group_balanced_entropy/**`
  - `benchmarks/results/mix_interleaved_probe/**`
  - `benchmarks/results/project_smoke/**`
  - `benchmarks/results/smoke/**`
- `pyproject.toml`
- `uv.lock`

## Required Validation Commands
- `uv lock --check`
- `uv run --group dev pytest -q`
- `python -m py_compile benchmarks/sdk_first_benchmark.py benchmarks/reference_strategy_benchmark.py`

## Required Artifact Checks
- Confirm no `.ipynb` files remain in the repository tree.
- Confirm no `experiment_runs.csv` remains in the repository tree.
- Confirm `pyproject.toml` and `uv.lock` do not contain `ipykernel`, `nbclient`, or `nbformat`.
- Confirm these accepted artifact directories exist and contain their expected summary/validation files:
  - `benchmarks/results/reference_full`
  - `benchmarks/results/class_group_balanced_entropy`
  - `benchmarks/results/mix_interleaved_probe`
  - `benchmarks/results/project_smoke`
  - `benchmarks/results/smoke`

## Summary File Requirements
`benchmarks/results/stage0_baseline_summary.md` should include:
- timestamp/date of validation;
- exact commands run and pass/fail outputs;
- accepted artifact directories and their key files;
- key benchmark facts already present in artifacts, without inventing new metrics;
- explicit note that notebook benchmark entrypoints and notebook dependencies are removed from the core baseline;
- concise residual limitations that belong to later stages, such as missing CoreSet/BADGE production implementations.

## Important Constraints
- Keep the file concise and factual.
- Do not claim Stage 1+ capabilities are implemented.
- If a validation command fails, stop and report the failure instead of writing a misleading baseline.

## Forbidden Actions
- Do not edit generated benchmark artifacts to make checks pass.
- Do not run destructive git commands.
- Do not modify unrelated dirty worktree files.

## Execution Plan
1. Run required validations.
2. Inspect accepted artifact directories and relevant summary JSON/MD files.
3. Write `benchmarks/results/stage0_baseline_summary.md` only if validations pass.
4. Report the new file and validation results.

## Acceptance Criteria
- The summary file exists and is accurate.
- All required validation commands pass.
- The summary can be used as Stage 0 closeout evidence before Stage 1 begins.

## Dependencies
- Should run after W48 dependency cleanup.
- Can run in parallel with R64 because R64 is read-only and does not inspect this new file as part of W49 review.
