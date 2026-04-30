# R29 - Review W19 Benchmark Fixes

## Relation To Overall Task
W19 fixed R28's required findings in the new benchmark harness:
- acquisition label leakage through provider metadata;
- non-strict JSON with NaN values.

This review gates acceptance of the benchmark harness before SDK algorithm improvements begin.

## Assumptions And Resolved Ambiguities
- R28 findings were valid and should be closed.
- W19 was allowed to modify `benchmarks/**` and smoke artifacts only.
- Existing unrelated dirty files outside `benchmarks/**` may predate this task; do not treat them as W19 changes.

## Goal And Expected Result
Perform read-only review of W19 changes and decide whether R28 blocking issues are fixed. Explicitly state whether there are no remaining in-scope defects, blocking risks, blocking questions, or required improvement requests.

## Responsibility Boundaries
Read-only review. Do not edit files.

## In Scope
- Verify acquisition-visible `DataSample.meta` no longer includes true labels.
- Verify labels are still available only in benchmark-private state for training/evaluation/diagnostics.
- Verify JSON artifacts do not contain literal `NaN`/`Infinity` and are strict JSON-compatible.
- Verify smoke artifacts exist and remain coherent.
- Re-check benchmark methodology for any new blocking regressions introduced by the fix.

## Out Of Scope
- SDK algorithm changes.
- Optional timing split/performance cleanup from R28.
- Product README rewrite.
- Dependency cleanup.

## Files Or Modules May Be Changed
None.

## Files Or Areas Must Not Be Touched
Entire repository.

## Important Architectural Constraints And Forbidden Actions
- Do not require reintroducing notebooks.
- Do not require full `ActiveLearningProject.run(...)` until SDK exposes public initial-label import.
- Keep optional follow-ups separate from blocking findings.

## High-Level Execution Plan
- Inspect `benchmarks/sdk_first_benchmark.py`, `benchmarks/README.md`, and `benchmarks/results/smoke/*.json`.
- Optionally run cheap read-only validations.
- Report whether R28 P1/P2 are closed.

## Acceptance Criteria
- R28 P1 is closed or a remaining leakage path is identified.
- R28 P2 is closed or a strict JSON defect is identified.
- Any new blocking problem is concrete and evidence-based.

## Expected Tests And Validations
Optional:
- `python -m py_compile benchmarks/sdk_first_benchmark.py`
- `python benchmarks/sdk_first_benchmark.py --help`
- strict JSON parse of `manifest.json` and `summary.json`

## Dependencies
Depends on W19.

## Parallel Or Sequential Notes
Sequential gate before benchmark acceptance.
