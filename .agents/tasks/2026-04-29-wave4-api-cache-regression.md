# Task: wave4-api-cache-regression

## Context
Repeat black-box SDK stress after another supposed fix round. Source inspection remains forbidden.

## Goal
Re-test Wave3 open issues (`get_round` raw `KeyError`, cache observability) and search for new public API/cache/state defects.

## Responsibility Boundaries
Owns only `.agents/tmp/blackbox_stress/wave4_api_cache/` and `.agents/tmp/blackbox_stress/wave4_api_cache_findings.md`.

## In Scope
- Read public docs and prior black-box reports.
- Use public SDK imports/API only.
- Re-test:
  - `ActiveLearningProject.get_round("unknown")`;
  - cache stats after prediction-heavy runs;
  - cache clear/reopen behavior;
  - previous fixed issues as spot checks.
- Search new issues in `get_state`, `validate`, `list_rounds`, `export_*`, `generate_report`, annotation timeout/review states, repeated attach/runtime rebinding, and public exception taxonomy.

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
- At least 35 API/cache cases attempted.
- Explicit old/open regression table.
- At least 10 new cases beyond Wave3.
- Findings include exact commands and expected/observed behavior.
