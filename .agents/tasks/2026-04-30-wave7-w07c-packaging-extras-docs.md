# W07C - Packaging, Extras, And Docs Smoke

## Task Identifier

W07C-PACKAGING-EXTRAS-DOCS

## Context

The SDK claims root import laziness, optional extras, wheel/sdist packaging, README quickstart behavior, and optional adapter import behavior. This subtask stress-tests install and documentation promises as a user would.

## Goal

Find packaging, optional dependency, README command, root import, adapter import, build, and metadata defects.

## Responsibility Boundaries

May write only:
- `.agents/tmp/blackbox_stress_wave7/w07c_packaging/**`

Must not touch:
- `src/active_learning_sdk/**`
- `tests/**`
- `benchmarks/*.py`
- project package metadata, lockfiles, or docs.

## In Scope

- `uv build` or equivalent package build.
- Fresh virtual environments installed from local wheel/sdist/editable package.
- Root import without optional dependencies.
- Optional extras import smoke where bounded.
- README quickstart copied as public user code, adjusted only for output paths.
- `pip check`, package metadata, wheel contents at a high level.

## Out Of Scope

- Inspecting SDK source inside built artifacts.
- Running full benchmark matrices.
- Live external services.

## Architectural Constraints

Use install/import behavior only. Do not read SDK implementation files from repo or from site-packages.

## Special Attention

Differentiate packaging bugs from environment limitations. Capture Python version, pip/uv command, and exact stderr/stdout snippets for failures.

## Execution Plan

1. Read README requirements/install sections and pyproject metadata.
2. Build or use existing wheel cautiously, recording command.
3. Create fresh venv(s) under task output directory.
4. Test core install, root import, README simulator quickstart, extras import smoke.
5. Write logs, `results.json`, and `findings.md`.

## Acceptance Criteria

- At least one clean core-install smoke in an isolated venv.
- README quickstart or documented equivalent executed.
- Optional extras behavior checked or explicitly skipped with reason.

## Dependencies

Can run in parallel with W07A/W07B/W07D.
