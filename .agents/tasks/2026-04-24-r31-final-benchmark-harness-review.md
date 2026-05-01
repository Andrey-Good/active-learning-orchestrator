# R31 - Final Benchmark Harness Review

## Relation To Overall Task
W21 addressed the final benchmark label-leakage finding from R30 by making IDs/groups label-independent and adding validation artifacts. This review decides whether the new benchmark harness is accepted.

## Assumptions And Resolved Ambiguities
- Previous required issues:
  - R28 P2 strict JSON should remain fixed.
  - R29/R30 label leakage through metadata, IDs, groups, and label-ordered ranges should now be fixed.
- W21 was allowed to edit `benchmarks/**` and smoke artifacts only.

## Goal And Expected Result
Perform final read-only review of the benchmark harness and smoke artifacts. Explicitly state whether there are no remaining in-scope defects, blocking risks, blocking questions, or required improvement requests.

## Responsibility Boundaries
Read-only. Do not edit files.

## In Scope
- Review `benchmarks/sdk_first_benchmark.py`.
- Review `benchmarks/README.md`.
- Review `benchmarks/results/smoke/**`, especially `validation.json`.
- Verify no acquisition-visible oracle labels leak through `meta`, `schema`, `sample_id`, `group_id`, or sorted ID/group ranges.
- Verify JSON strictness and artifact coherence.
- Verify benchmark still uses SDK `StrategyScheduler`/`SelectionContext` for acquisition.

## Out Of Scope
- SDK algorithm changes.
- Full benchmark run.
- Product README rewrite.
- Dependency cleanup.
- Optional timing split/performance improvements.

## Files Or Modules May Be Changed
None.

## Files Or Areas Must Not Be Touched
Entire repository.

## Important Architectural Constraints And Forbidden Actions
- Do not require labels to be absent from synthetic text or post-selection diagnostic label distributions.
- Do not require full `ActiveLearningProject` loop before initial-label SDK API exists.
- Separate optional follow-ups from blockers.

## High-Level Execution Plan
- Inspect code and smoke artifacts.
- Run cheap validations if useful.
- Report final benchmark acceptance status.

## Acceptance Criteria
- No remaining oracle leakage path in acquisition-visible fields/ranges.
- Strict JSON remains valid.
- Smoke artifacts are coherent and complete.
- Final answer explicitly says whether benchmark harness can be used for SDK optimization.

## Expected Tests And Validations
Optional:
- `python -m py_compile benchmarks/sdk_first_benchmark.py`
- `python benchmarks/sdk_first_benchmark.py --help`
- strict JSON parse/no NaN check
- inspect/read `validation.json`

## Dependencies
Depends on W21.

## Parallel Or Sequential Notes
Sequential gate before SDK optimization.
