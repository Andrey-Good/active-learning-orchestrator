# Task: wave3-regression-and-new-api

## Context
The user requested repeat black-box stress testing after the previously reported issues were supposedly fixed. SDK source inspection remains forbidden.

## Goal
Re-test the prior API/model/cache contract failures and then search for new public API/state/adapter defects.

## Responsibility Boundaries
Owns only artifacts under `.agents/tmp/blackbox_stress/wave3_api/` and findings in `.agents/tmp/blackbox_stress/wave3_api_findings.md`.

## In Scope
- Read public documentation and prior black-box reports.
- Run existing black-box reproduction harnesses if useful.
- Write new external consumer-style tests under the owned directory.
- Verify prior API issues:
  - custom selector duplicate IDs;
  - custom selector wrong exception category;
  - missing `data["text"]`;
  - stochastic/committee shape contract behavior;
  - cache stats/persistence inconsistency if feasible.
- Search for new issues in public project methods, attach/resume, import/export, report generation, custom backend behavior, prelabeling, split modes, adapter capability inspection, and repeated runs.

## Out of Scope
- Reading `src/active_learning_sdk/**`.
- Modifying SDK source, tests, benchmarks, docs, or sibling stress directories.
- Fixing defects.

## Must Not Touch
- `src/**`
- `tests/**`
- `benchmarks/**` except by public documented command execution if needed
- `.agents/tmp/blackbox_stress/wave3_*` directories owned by other workers

## Execution Plan
1. Create or reuse black-box harnesses under the owned directory.
2. First classify old findings as fixed/still-open/changed.
3. Add new adversarial cases beyond old reproductions.
4. Record exact commands, observed behavior, expected behavior, severity, and artifact paths.

## Acceptance Criteria
- At least 30 cases attempted.
- Explicit old-finding regression table.
- At least 10 new cases beyond the old reproductions.
- No SDK source inspection.
