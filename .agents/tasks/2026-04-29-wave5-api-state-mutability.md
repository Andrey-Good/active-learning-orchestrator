# Task: wave5-api-state-mutability

## Context
Repeat black-box SDK stress after another claimed fix round. Source inspection remains forbidden.

## Goal
Re-test Wave4 API/state issues and search for new state-integrity defects.

## Responsibility Boundaries
Owns only `.agents/tmp/blackbox_stress/wave5_api_state/` and `.agents/tmp/blackbox_stress/wave5_api_state_findings.md`.

## In Scope
- Read public docs and prior black-box reports.
- Use public SDK APIs only.
- Re-test:
  - `get_state()` copy-safety / immutability;
  - `get_round()` copy-safety;
  - `list_rounds()` copy-safety;
  - cache stats/clear/reopen spot checks;
  - public exception taxonomy.
- Search new state bugs: nested mutable structures in state/status/report/export, mutation persistence after close/reopen, stale state after attach, validate/status consistency after returned-object tampering, clear-cache metadata mutation, round payload mutation.

## Out of Scope
- Reading `src/active_learning_sdk/**`.
- Modifying SDK source/tests/docs/benchmarks.
- Fixing bugs.

## Must Not Touch
- `src/**`
- `tests/**`
- benchmark source files
- sibling wave5 directories

## Acceptance Criteria
- At least 45 API/state cases attempted.
- Explicit regression table for Wave4 issues.
- At least 12 new cases beyond Wave4.
- Findings include exact reproduction command and severity.
