# W08C Packaging, Docs, Extras, And CLI Smoke

Task identifier: `wave8-w08c-packaging-docs-cli`

## Goal

Validate the project as a package and documented command-line/user-facing artifact without reading implementation source.

## Ownership

May write only under `.agents/tmp/blackbox_stress_wave8/w08c_packaging_docs/`.

Must not touch product files, docs, tests, benchmark source, or other workers' directories.

## In Scope

- `uv build`, `twine check`, and isolated wheel/sdist install checks.
- Root import without optional dependencies.
- README core simulator quickstart.
- Optional extras probes for `sklearn`, `xxhash`, `datasets`, `benchmarks`, and bounded `huggingface` scaffold behavior if feasible.
- Documented benchmark CLI commands may be executed as black-box CLIs, but benchmark implementation source must not be read.
- Package metadata consistency with README claims.

## Out Of Scope

- Reading implementation source under `src/**`.
- Reading repository tests under `tests/**`.
- Reading benchmark implementation source under `benchmarks/*.py`.
- Full heavy `all` extra if it would consume excessive time; if skipped, record as coverage gap, not defect.

## Plan

1. Create isolated temp directories/virtual environments under the owned tmp directory.
2. Build artifacts and run metadata validation.
3. Install core artifact and run import plus README quickstart.
4. Probe optional extras with small import/runtime checks.
5. Execute at least one documented benchmark preset if feasible.
6. Write `findings.md`, `results.json`, and key command logs.

## Acceptance Criteria

- Distinguish actual packaging/doc failures from skipped heavy optional coverage.
- Include exact commands and exit codes for every issue.
