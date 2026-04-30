# Task: wave5-backend-adapter-exploration

## Context
Repeat black-box exploration after another claimed fix round. Prior Wave4 accepted simulator same-round/different-sample-set and hostile property capability-inspection issues.

## Goal
Re-test Wave4 exploratory issues and search for new backend/adapter/prelabel/package defects.

## Responsibility Boundaries
Owns only `.agents/tmp/blackbox_stress/wave5_explore/` and `.agents/tmp/blackbox_stress/wave5_explore_findings.md`.

## In Scope
- Read public docs, package metadata, prior reports.
- Use public imports and user-side test doubles only.
- Re-test:
  - simulator same round id with different sample set;
  - `inspect_model_capabilities()` hostile property behavior.
- Search new issues in simulator idempotency, annotation policies, custom backend pull/push edge cases, prelabel thresholds/payloads, sklearn adapter, packaging/docs drift, optional backend imports.

## Out of Scope
- Reading `src/active_learning_sdk/**`.
- Modifying SDK/tests/docs/benchmarks.
- Fixing bugs.

## Must Not Touch
- `src/**`
- `tests/**`
- benchmark source files
- sibling wave5 directories

## Acceptance Criteria
- At least 30 exploratory cases attempted.
- At least one backend-hostility case, one adapter/capability case, one prelabel case, one packaging/docs case.
- Explicit regression table for Wave4 exploratory issues.
