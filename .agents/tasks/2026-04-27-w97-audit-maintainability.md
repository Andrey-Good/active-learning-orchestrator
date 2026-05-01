# Task W97-D: Maintainability and Code-Smell Senior Audit

## Context
The user is specifically worried about AI-generated repository rot: hidden hacks, duplicated logic, unclear contracts, stale docs, dead artifacts, and code that will be hard to maintain in one or two years.

## Goal
Perform a read-only maintainability audit. Focus on long-term clarity, dependency hygiene, packaging, public API stability, and documentation consistency.

## Ownership
Read-only scope:
- Entire repository
- `README.md`
- `pyproject.toml`
- `docs/**`
- `docker/**`
- Public exports in `src/active_learning_sdk/**`

Do not edit files.

## In Scope
- Overly large files/functions and tangled responsibilities.
- Stale generated artifacts or outdated audit evidence.
- Missing architectural notes for invented logic.
- Dependency risks and packaging issues.
- Public API inconsistencies.
- Tests that encode implementation accidents rather than contract.

## Out of Scope
- Deep mathematical verification of strategies.
- Running long benchmarks.

## Special Attention
- Claims in README/docs that current code does not support.
- Deprecated files or duplicate AGENT/AGENTS naming issues.
- Untracked benchmark artifacts that should not be shipped.
- Hidden reliance on notebooks or local state.

## Expected Output
- Findings ordered by severity.
- Specific cleanup/fix recommendations.
- Whether each issue is release-blocking or a post-release maintainability task.
