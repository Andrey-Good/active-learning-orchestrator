# Stage 2B: Native External Active-Learning Benchmark Entry Point

## Context

The existing reference harness has `modal_formula_*` and `skactiveml_formula_*` rows. They are honest formula shims but do not call native external query APIs. Stage 2 requires a separate opt-in native workflow benchmark, with skipped rows when optional libraries are unavailable.

Primary API references reviewed by the orchestrator:

- modAL uncertainty API: `modAL.uncertainty.entropy_sampling`, `margin_sampling`, and `uncertainty_sampling` return selected indices and selected instances.
- scikit-activeml API reference: pool query strategies include `UncertaintySampling` and `QueryByCommittee`.

## Goal

Add an opt-in native external benchmark entrypoint that does not run by default and does not add external libraries as mandatory dependencies.

## Ownership

You may edit:

- `benchmarks/native_external_benchmark.py`
- `benchmarks/README.md`
- `tests/test_native_external_benchmark.py`
- `pyproject.toml` only for optional benchmark extras if needed

Do not edit SDK runtime, existing reference formula code except docs links if needed, or old artifacts.

## In Scope

- Add a small CLI script with `--preset smoke`, `--libraries`, `--strategies`, `--output-dir`, `--overwrite`.
- Implement native rows defensively:
  - `modal_native_entropy`, `modal_native_margin`, `modal_native_least_confidence` using modAL uncertainty sampling if importable.
  - `skactiveml_native_uncertainty` or equivalent using scikit-activeml if importable.
- If a library/API is unavailable or incompatible, record a skipped row with a clear reason rather than failing the whole benchmark.
- Emit strict JSON/CSV/Markdown artifacts:
  - `native_external_results.csv`
  - `native_external_summary.json`
  - `manifest.json`
  - `summary.md`
- Include reproducibility metadata and explicit claim category `native_external_workflow_smoke`.
- Add tests using monkeypatched fake external modules so the native paths are exercised without installing optional dependencies.

## Out Of Scope

- Large real-dataset external benchmark.
- Claiming performance superiority.
- Making modAL/scikit-activeml required dependencies.

## Constraints

- Keep the script importable when optional external libraries are missing.
- Keep smoke runtime tiny.
- Be explicit that this is a native-query smoke, not a full end-to-end production comparison.

## Suggested Validation

- `uv run pytest tests/test_native_external_benchmark.py -q`
- Run the script with missing external libs and confirm skipped artifacts.
- `uv run pytest -q`
- `uv run --with ruff ruff check .`

## Acceptance Criteria

- There is a separate native external benchmark entrypoint.
- Missing optional libraries produce evidence artifacts with skip reasons.
- Fake-module tests prove native API calls are wired.
