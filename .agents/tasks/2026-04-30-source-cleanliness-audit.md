# Task: 2026-04-30 Source Cleanliness Audit

## Context
The repository is functionally green, but the user requested a release-quality cleanup pass that must not change SDK behavior or benchmark metrics.

## Goal
Find source-code cleanliness issues that can be safely improved without changing behavior.

## Responsibility Boundaries
Own read-only review of `src/active_learning_sdk/**`.

## In Scope
- Dead code candidates.
- Duplicated helper logic.
- Obvious no-op code, unused imports, unreachable branches.
- Hacky comments, temporary scaffolding, misleading names.
- Places where code can be simplified without behavior change.

## Out of Scope
- Editing files.
- Algorithm changes.
- Public API changes.
- Test rewrites.
- Benchmark or README edits.

## Must Not Touch
- Do not modify any file.
- Do not run destructive commands.
- Do not propose risky refactors that could alter active-learning behavior or metrics.

## Acceptance Criteria
- Report findings grouped by confidence and risk.
- Mark each candidate as safe/no-safe.
- Include exact file paths and line anchors where possible.
- Prefer "no change" if a cleanup is not obviously behavior-preserving.

## Expected Validation
For any proposed cleanup, specify which existing tests should cover it and whether extra regression tests are needed.
