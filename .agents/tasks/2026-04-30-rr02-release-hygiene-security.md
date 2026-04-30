# 2026-04-30 rr02 Release Hygiene And Security Audit

## Task Identifier

`rr02-release-hygiene-security`

## Context

This is part of the public GitHub release-readiness assessment for the current SDK worktree. Unlike the black-box runtime subtask, this subtask may inspect repository files because public release safety requires checking metadata, license, secrets, generated artifacts, and accidental private material.

## Goal

Identify repository-level reasons the SDK should not be published publicly yet.

## Responsibility Boundaries

Read-only audit. Do not edit files. Do not clean the worktree.

## In Scope

- Git status and public repository shape.
- License and packaging metadata.
- `.gitignore` coverage and risky generated files.
- Potential secrets, credentials, tokens, private URLs, internal-only names, local absolute paths, notebooks, datasets, caches, and build artifacts.
- Public GitHub URL consistency and release workflow readiness.

## Out of Scope

- Fixing secrets or rewriting history.
- Publishing or creating releases.
- Deep algorithmic correctness review.

## Files Or Areas May Be Changed

- None.

## Files Or Areas Must Not Be Touched

- Entire repository.

## Special Attention

- Use fast automated scans and targeted reads.
- Do not report false positives as blockers without checking context.
- Treat untracked files as part of what could accidentally be released if not curated.

## Execution Plan

1. Review release metadata and repository state.
2. Scan for obvious secret/leak indicators.
3. Inspect package inclusion/exclusion risk.
4. Produce prioritized findings.

## Acceptance Criteria

- Release-blocking repository hygiene/security risks are clearly separated from nice-to-have cleanup.
