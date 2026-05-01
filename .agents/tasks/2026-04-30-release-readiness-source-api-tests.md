# 2026-04-30 Release Readiness: Source, API, And Tests

## Task Identifier

release-readiness-source-api-tests

## Context

Assess whether the current code and validation surface are mature enough for public GitHub release. This is a white-box complement to the black-box package/docs review.

## Goal

Check public API coherence, implementation/test health, CI expectations, type/lint/test/build status, and obvious source-level release risks.

## Responsibility Boundaries

In scope:

- `src/active_learning_sdk/`, `tests/`, `.github/workflows/`, `pyproject.toml`, CI/test commands.
- Running local validation commands if practical.
- Looking for API/docs mismatch and test gaps that affect release readiness.

Out of scope:

- Secret scanning as the primary responsibility.
- Editing source files.
- Publishing or uploading artifacts.

## Files Or Areas That May Be Read

- `src/`
- `tests/`
- `.github/`
- `pyproject.toml`
- README/docs only as needed to check API mismatch.

## Files Or Areas That Must Not Be Touched

- Do not edit any file.
- Do not revert dirty working tree changes.

## Special Attention

- Whether README-claimed public APIs are exported/importable.
- Whether validation commands pass in the current worktree, not only in stale docs.
- Whether public release metadata matches current module/package names.
- Whether CI is sufficient for Python version and optional extras.

## Acceptance Criteria

- Report validation commands run and exact pass/fail status.
- Provide concise blocker/high/medium/low findings with evidence.
- Avoid speculative findings without reproduction.

## Dependencies

None. Can run in parallel with the other release-readiness reviews.
