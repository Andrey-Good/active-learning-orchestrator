# R71 - Review Stage 1 Public Sklearn Project Smoke

## Context
W56 updated `project_smoke` so it exercises the public `SklearnTextClassifierAdapter` instead of a benchmark-local adapter.

## Goal
Review whether project smoke now proves the Stage 1 adapter baseline through the public SDK loop without benchmark-only hacks.

## Responsibility Boundaries
- This is a read-only review.
- Focus on W56-owned benchmark/test changes.

## In Scope
- `benchmarks/sdk_first_benchmark.py`
- `tests/test_project_smoke_benchmark.py`
- CLI smoke behavior using a temporary output directory.

## Out of Scope
- Do not edit files.
- Do not regenerate committed benchmark artifacts.
- Do not review unrelated benchmark strategy science.

## Review Questions
- Does `run_project_smoke` instantiate and use `active_learning_sdk.adapters.SklearnTextClassifierAdapter`?
- Are benchmark-only adapter hacks removed from the public project smoke path?
- Are assertions based on public project evidence, not private engine/state mutation?
- Does the test use a temp output directory and avoid touching `benchmarks/results/**`?
- Does CLI `--preset project_smoke` pass quickly?
- Do full tests pass?

## Validation
- `uv run --group dev pytest -q tests/test_project_smoke_benchmark.py`
- `uv run python benchmarks/sdk_first_benchmark.py --preset project_smoke --output-dir <temp path>`
- `uv run --group dev pytest -q`

## Forbidden Actions
- Do not edit files.
- Do not run destructive git commands.

## Acceptance Criteria
- No findings remain.
- Public project smoke is acceptable as Stage 1 exit evidence.
