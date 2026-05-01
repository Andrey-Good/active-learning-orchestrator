# 2026-04-28-w09-test-suite-coverage-hygiene-objection-sweep

## Context

Second-pass exhaustive senior audit requested on 2026-04-28.

## Goal

Audit test-suite quality, CI/dev tooling assumptions, repo hygiene, generated artifact sprawl, untracked critical files, ignored curated evidence, and blind spots in existing tests.

## Responsibility Boundaries

Owner may change only:

- `.agents/tmp/2026-04-28-w09-test-suite-coverage-hygiene-findings.md`

Owner must not change:

- `src/**`
- tests
- benchmarks
- docs
- dependency files

## In Scope

- test inventory and coverage themes
- lint/type/build commands
- git tracked vs untracked/ignored artifacts
- generated caches
- task/doc sprawl
- missing CI config if applicable

## Out of Scope

- Writing new tests.
- Fixing repo hygiene.

## Constraints

- Read-only.
- Use PowerShell fallback if `rg` fails.
- Findings must be actionable and not duplicate earlier W01-W04 findings unless adding new evidence.

## Acceptance Criteria

- Identify real maintainability/release risks.
- Include severity and remediation.
- Avoid complaining about existing dirty worktree as if created by this audit.

## Dependencies

Can run in parallel with W05, W06, W07, and W08.
