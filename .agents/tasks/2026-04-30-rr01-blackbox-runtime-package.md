# 2026-04-30 rr01 Black-Box Runtime And Package Audit

## Task Identifier

`rr01-blackbox-runtime-package`

## Context

This is part of the public release-readiness assessment for the current SDK worktree. Use the `project-black-box-stress-test` posture for this subtask: behave like an external SDK user and rely on documentation, public examples, package/runtime behavior, logs, and generated artifacts.

## Goal

Determine whether the SDK can be installed, imported, built, and used through its documented public quickstart without source-code knowledge.

## Responsibility Boundaries

Read-only audit. Do not edit files. You may create temporary test scripts or virtual environments only under `.agents/tmp/release_readiness_2026_04_30/rr01`.

## In Scope

- `README.md`, packaging metadata, public examples, build artifacts, wheel/sdist behavior.
- Build/install/import smoke tests.
- README simulator quickstart or a close documented equivalent.
- Externally observable errors and developer-experience blockers.

## Out of Scope

- Inspecting implementation source files.
- Fixing defects.
- Long benchmark campaigns.

## Files Or Areas May Be Changed

- Temporary scratch files under `.agents/tmp/release_readiness_2026_04_30/rr01` only, if needed.

## Files Or Areas Must Not Be Touched

- `src/`, `tests/`, `docs/`, `README.md`, `pyproject.toml`, `uv.lock`, `dist/`.

## Special Attention

- Validate the current worktree, not a remembered older state.
- Separate packaging/docs problems from runtime bugs.
- Every confirmed issue needs reproduction path, expected behavior, actual behavior, severity, confidence, and evidence.

## Execution Plan

1. Build a test map for public-user package behavior.
2. Run fast external-user checks.
3. Report confirmed release blockers and non-blocking risks.

## Acceptance Criteria

- Findings are reproducible and evidence-backed.
- No source inspection is used for this black-box subtask.
