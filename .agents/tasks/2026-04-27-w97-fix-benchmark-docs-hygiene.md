# Task W97-G: Fix Benchmark Evidence and Release Hygiene Issues

## Context
Senior maintainability/benchmark audits found that release evidence and docs are too strong or brittle:
- reference benchmark smoke was broken until W97 local fix;
- README overclaims general superiority and safest defaults;
- benchmark result directories are uncontrolled;
- benchmark manifests lack reproducibility metadata;
- SDK-first benchmark overwrites output directories by default;
- dependency packaging is too heavy for core install.

## Goal
Make benchmark evidence and docs safer and more honest without deleting user work.

## Ownership
Allowed write scope:
- `benchmarks/sdk_first_benchmark.py`
- `benchmarks/README.md`
- `README.md`
- `.gitignore`
- `pyproject.toml` only if dependency extras can be changed safely
- focused benchmark tests under `tests/`

Do not edit:
- SDK runtime/state/strategy implementation.
- Existing generated benchmark outputs except newly generated W97 outputs.

## Requirements
- Add reproducibility metadata to benchmark manifests where feasible: argv, git SHA/dirty, python version, platform, artifact schema version.
- Prevent accidental overwrite of benchmark output dirs unless explicit overwrite is requested, or default to run-id directories.
- Downgrade README claims from broad production superiority to scoped diagnostic evidence.
- Move generated results toward ignored-by-default or clearly curated evidence.
- Be conservative with dependency changes; do not break current `uv run pytest`/`uv build`.

## Acceptance
- Benchmark smoke commands still run.
- Tests cover no-clobber/manifest metadata if practical.
- README no longer overstates benchmark evidence.
