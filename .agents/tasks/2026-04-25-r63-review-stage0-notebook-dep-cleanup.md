# R63 - Review Stage 0 Notebook Dependency Cleanup

## Context
Stage 0 stabilizes the current SDK core before any new active-learning capabilities are added. Worker W48 removed notebook-only dependencies from the core dependency set.

## Goal
Review W48 changes for correctness, lockfile consistency, and absence of accidental scope expansion.

## Responsibility Boundaries
- This is a read-only review task.
- Inspect only dependency and validation-relevant state.

## In Scope
- Review `pyproject.toml`.
- Review `uv.lock`.
- Verify `ipykernel`, `nbclient`, and `nbformat` are no longer project dependencies.
- Verify the lockfile is consistent with `pyproject.toml`.
- Run or inspect validation results:
  - `uv run --group dev pytest -q`
  - `python -m py_compile benchmarks/sdk_first_benchmark.py benchmarks/reference_strategy_benchmark.py`

## Out of Scope
- Do not edit files.
- Do not add or remove dependencies.
- Do not touch source code, benchmarks, README, or task docs.

## Files That May Be Changed
- None.

## Files That Must Not Be Touched
- All files.

## Important Constraints
- The SDK core must not depend on notebook runtimes.
- Benchmark scripts must remain executable Python modules, not notebook-dependent workflows.
- Existing dirty worktree changes unrelated to W48 must not be reverted.

## Forbidden Actions
- Do not run destructive git commands.
- Do not modify or regenerate files.
- Do not broaden the review into unrelated roadmap stages.

## Execution Plan
1. Inspect dependency files for the removed notebook dependencies.
2. Confirm lockfile consistency by checking whether `uv lock --check` is available or by running an equivalent non-mutating validation.
3. Run the Stage 0 validation commands if feasible.
4. Report findings with file and line references if any.

## Acceptance Criteria
- No notebook-only dependencies remain in core dependency declarations or lockfile.
- Validation commands pass.
- No unrelated changes are attributed to W48.

## Dependencies
- Depends on W48 completion.

## Notes
This review gates the next Stage 0 documentation baseline cleanup task.
