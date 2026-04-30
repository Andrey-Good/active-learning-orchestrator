# W56 - Stage 1 Public Sklearn Adapter Project Smoke

## Context
Stage 1 exit criteria require the new sklearn adapter to work through the public SDK project loop, not only in isolated adapter tests. The benchmark `project_smoke` path currently uses a benchmark-local `SklearnTextBenchmarkAdapter`.

## Goal
Make the public project smoke benchmark exercise `active_learning_sdk.adapters.SklearnTextClassifierAdapter` directly, and add/adjust tests so this remains covered.

## Responsibility Boundaries
Own only public project smoke integration for the sklearn adapter.

## In Scope
- `benchmarks/sdk_first_benchmark.py`
- New or existing tests under `tests/` that validate the project smoke path uses the public sklearn adapter.
- If strictly necessary, tiny adapter test adjustments in `tests/test_sklearn_adapter.py`.

## Out of Scope
- Do not edit core capability contracts.
- Do not edit `src/active_learning_sdk/adapters/sklearn.py` unless the benchmark exposes a real adapter bug; if so, report before changing.
- Do not edit README or roadmap docs.
- Do not regenerate committed benchmark result artifacts.
- Do not run long benchmark presets.

## Required Behavior
- `run_project_smoke(...)` must use `SklearnTextClassifierAdapter` from the SDK public adapter module.
- Avoid benchmark-only model warm-starts or benchmark-only adapter subclasses for the project smoke path.
- Keep the benchmark-local `SklearnTextBenchmarkAdapter` only if still needed by scheduler-level strategy curves; do not broaden refactors.
- Replace `fit_count`-based project smoke assertions with public-SDK-observable evidence, such as:
  - public run_step sequence includes seed `train_eval`;
  - completed active round includes metrics after training;
  - model id changes after training if using `get_model_id()`;
  - project state metrics history contains expected seed/active evaluation records.
- Ensure `project_smoke` artifacts explicitly state that the public sklearn adapter is used.

## Test Requirements
- Add or update tests so `project_smoke` can be run in a temp output dir with the public sklearn adapter and completes one active round quickly.
- The test should not download data or call external services.
- The test should not rely on notebooks.

## Validation
- Run the new/updated project smoke test.
- Run `uv run python benchmarks/sdk_first_benchmark.py --preset project_smoke --output-dir <temp or ignored path>`.
- Run `uv run --group dev pytest -q`.

## Files That Must Not Be Touched
- `src/active_learning_sdk/adapters/base.py`
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/strategies/**`
- `README.md`
- `pyproject.toml`
- `uv.lock`
- `benchmarks/results/**`

## Forbidden Actions
- Do not run destructive git commands.
- Do not modify accepted benchmark artifacts.
- Do not revert unrelated dirty worktree changes.

## Acceptance Criteria
- The public project smoke benchmark uses `SklearnTextClassifierAdapter`.
- Full tests pass.
- A one-command project smoke benchmark run passes in under a few minutes.
