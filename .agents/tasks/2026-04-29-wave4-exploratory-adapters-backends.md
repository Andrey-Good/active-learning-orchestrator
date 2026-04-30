# Task: wave4-exploratory-adapters-backends

## Context
Repeat black-box stress after another supposed fix round. The API/cache and quality workers cover old failures; this task looks for new defects in less-hit public/provisional surfaces.

## Goal
Stress adapters, backends, packaging/docs, and edge-case user implementations from a black-box consumer perspective.

## Responsibility Boundaries
Owns only `.agents/tmp/blackbox_stress/wave4_explore/` and `.agents/tmp/blackbox_stress/wave4_explore_findings.md`.

## In Scope
- Read public docs and package metadata.
- Use public imports and optional modules as a user would.
- Test sklearn adapter behavior if dependency is present.
- Test simulator/custom backend hostile cases: duplicate annotations, conflicting labels, empty pulls, wrong round ids, delayed readiness.
- Test prelabel payload shapes and confidence thresholds.
- Test packaging quick checks if metadata changed.
- Probe docs examples for currently accurate behavior.

## Out of Scope
- Reading `src/active_learning_sdk/**`.
- Modifying SDK source/tests/docs/benchmarks.
- Fixing bugs.

## Must Not Touch
- `src/**`
- `tests/**`
- benchmark source files
- sibling wave4 directories

## Acceptance Criteria
- At least 25 exploratory cases attempted.
- At least one sklearn adapter case if available.
- At least one backend-hostility case, one prelabel case, one packaging/docs case.
- Findings include severity and reproduction commands.
