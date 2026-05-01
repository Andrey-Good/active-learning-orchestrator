# Task: 2026-04-30 Packaging And Release Hygiene Audit

## Context
The release-readiness report found packaging/publication blockers: wrong project URLs, broad sdist contents, `.agents` visibility, stale README evidence, and unclear publishable state.

## Goal
Identify packaging and repository hygiene changes that are safe and behavior-neutral.

## Responsibility Boundaries
Own read-only review of packaging metadata and release-visible files.

## In Scope
- `pyproject.toml` metadata, URLs, package data, sdist include/exclude.
- `.gitignore` coverage for internal/runtime/generated artifacts.
- `dist` contents, `twine check` implications, wheel/sdist artifact contents.
- GitHub workflow/repo-publication hygiene if present.

## Out of Scope
- Editing files.
- Source behavior changes.
- Test behavior changes.

## Must Not Touch
- Do not modify any file.
- Do not delete artifacts.
- Do not assume a real GitHub repository URL unless supported by local config/docs.

## Acceptance Criteria
- List release blockers versus optional cleanup.
- For each blocker, propose the smallest safe fix.
- Include validation commands after fixes.
- Explicitly state if a report finding is not real.
