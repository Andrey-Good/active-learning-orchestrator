# 2026-04-30 Release Readiness: Security And Repository Hygiene

## Task Identifier

release-readiness-security-hygiene

## Context

Assess whether the repository is safe to publish publicly on GitHub from a hygiene/security perspective.

## Goal

Find secrets, credentials, private data, oversized/generated artifacts, local-only files, confusing git state, or public-release hygiene issues.

## Responsibility Boundaries

In scope:

- Secret/token patterns in tracked and candidate public files.
- `.gitignore`, `.github/`, Docker configs, docs, benchmark outputs, task logs, notebooks, generated artifacts, `dist/`.
- Git status and obvious release metadata risks.

Out of scope:

- Deep behavioral source-code bug review.
- Uploading or publishing anything.
- Deleting or modifying files.

## Files Or Areas That May Be Read

- Repository files except `.venv/` and `.git/objects`.
- `.gitignore`, `.github/`, `docker/`, `docs/`, `benchmarks/`, `dist/`, root metadata files.

## Files Or Areas That Must Not Be Touched

- Do not edit any file.
- Do not reveal full secret values if found.

## Special Attention

- Placeholder credentials versus real credentials.
- Local task files and previous audit reports that may not belong in a polished public release.
- Whether `.venv`, local run directories, notebooks, or benchmark outputs could be committed accidentally.
- License/community/security file completeness.

## Acceptance Criteria

- Provide a prioritized list of confirmed hygiene/security issues.
- Include evidence paths and safe reproduction commands.
- Distinguish release blockers from cleanup recommendations.

## Dependencies

None. Can run in parallel with the other release-readiness reviews.
