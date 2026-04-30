Task ID: W48
Short name: Stage 0 remove notebook core dependencies
Relation to overall task: Stage 0 of the professional SDK roadmap. Stabilize current core by removing notebook-only dependencies from the core package install now that notebook entrypoints were removed.

Assumptions and resolved ambiguities:
- `.ipynb` entrypoints and legacy notebook benchmark artifacts are removed.
- `ipykernel`, `nbclient`, and `nbformat` are still in `[project.dependencies]`.
- The goal is a cleaner runtime/core install, not adding notebook support back.
- This worker is not alone in the codebase. Do not revert or overwrite unrelated edits.

Goal and expected result:
- Remove notebook-only dependencies from core package dependencies.
- Update the lockfile consistently.
- Verify tests still pass without code changes.

Responsibility boundaries:
- In scope:
  - `pyproject.toml`
  - `uv.lock`
- Out of scope:
  - `src/**`
  - `tests/**`
  - `README.md`
  - benchmark code/artifacts
  - docs

Architectural constraints and forbidden actions:
- Do not reintroduce notebooks.
- Do not add new dependencies unless strictly required.
- Do not hand-edit lockfile if `uv lock` can update it safely.
- Preserve unrelated dependency pins.

Execution plan:
- Remove `ipykernel==7.2.0`, `nbclient==0.10.4`, and `nbformat==5.10.4` from `[project.dependencies]`.
- Run `uv lock` to update `uv.lock`.
- Run focused validation:
  - `uv run --group dev pytest -q`
  - `python -m py_compile benchmarks/sdk_first_benchmark.py benchmarks/reference_strategy_benchmark.py`
- Search project-owned files, excluding `.venv`, for remaining core dependency references.

Acceptance criteria:
- Core dependencies no longer include notebook-only packages.
- Lockfile no longer lists those packages as direct project requirements.
- Tests pass.
- No source/test/benchmark implementation files are changed.

Expected validations:
- `uv lock`
- `uv run --group dev pytest -q`
- `python -m py_compile benchmarks/sdk_first_benchmark.py benchmarks/reference_strategy_benchmark.py`
- Search pyproject/lock for removed dependency names.

Dependencies:
- Starts Stage 0 implementation.

Parallel/sequential notes:
- Must be reviewed before Stage 0 baseline freeze.
