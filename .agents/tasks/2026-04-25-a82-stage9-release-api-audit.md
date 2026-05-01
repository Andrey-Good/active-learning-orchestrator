# A82 - Stage 9 Release API And Packaging Audit

## Context

Stage 9 is the final release-hardening pass. Before README is updated, we need a focused audit of packaging, public API exports, generated files, and release blockers.

## Goal

Identify concrete release-readiness gaps in the current SDK package and public API surface. This is an audit task only.

## Responsibility Boundaries

You are an auditor/reviewer. Do not edit files.

## In Scope

- `pyproject.toml`
- `src/active_learning_sdk/__init__.py`
- `src/active_learning_sdk/adapters/__init__.py`
- `src/active_learning_sdk/backends/__init__.py`
- `src/active_learning_sdk/strategies/__init__.py`
- `src/active_learning_sdk/backends/assets/**`
- `docker/label_studio/**`
- root `README.md`
- `.gitignore`
- package build behavior

## Out of Scope

- Benchmark execution/report writing
- SDK algorithm changes
- README rewriting

## Special Attention

- Does the package include required managed Label Studio assets in wheel and sdist?
- Are useful product classes exported coherently?
- Are stale or false README claims obvious?
- Are generated caches/build artifacts likely to pollute the release?
- Does `uv build` succeed?
- Does import from a built wheel work for main public modules?

## Forbidden Actions

- Do not modify files.
- Do not delete files.
- Do not revert unrelated workspace changes.

## Review Plan

1. Inspect package metadata and exports.
2. Run `uv build` to a temp output directory if feasible.
3. Inspect wheel/sdist contents for required assets and accidental large/runtime files.
4. Run import smoke from the installed project or built wheel if feasible.
5. Return findings first with severity and exact file references, followed by commands run.

## Acceptance Criteria

- Clear release blockers are identified.
- If no blockers are found, state residual risks.

## Dependencies

Can run in parallel with W82. Its findings inform Stage 9 hardening before README finalization.
