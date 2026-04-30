# 2026-04-28-w06-state-cache-report-export-objection-sweep

## Context

Second-pass exhaustive senior audit requested on 2026-04-28.

## Goal

Audit persisted state, cache stores, JSON safety, report generation, exports, validation coverage, and corruption recovery. Produce additional failing tests for concrete defects plus a findings note.

## Responsibility Boundaries

Owner may change only:

- `tests/test_objection_sweep_state_cache_report_2026_04_28.py`
- `.agents/tmp/2026-04-28-w06-state-cache-report-export-findings.md`

Owner must not change:

- `src/**`
- existing tests
- benchmark files
- docs except the owned findings file

## In Scope

- `state/store.py`
- `cache.py`
- `report.py`
- `utils.py`
- export APIs in `engine.py`
- validation behavior in `engine.py`

## Out of Scope

- Label backend network behavior.
- Strategy ranking.
- Packaging metadata.

## Constraints

- Tests must be deterministic and local.
- Avoid giant generated files.
- No production code edits.

## Acceptance Criteria

- Findings distinguish actual data-loss/corruption risks from maintainability concerns.
- Reproductions are small and bounded.
- Every objection has severity and remediation.

## Dependencies

Can run in parallel with W05, W07, W08, and W09.
