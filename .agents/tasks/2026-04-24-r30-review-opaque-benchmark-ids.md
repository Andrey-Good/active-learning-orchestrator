# R30 - Review Opaque Benchmark IDs Fix

## Relation To Overall Task
W20 fixed the remaining benchmark label leakage reported by R29 by making acquisition-visible sample IDs and group IDs opaque. This is the final gate before accepting the new benchmark harness and beginning SDK algorithm improvements.

## Assumptions And Resolved Ambiguities
- R29 P1 was valid and must be closed.
- R29 P2 strict JSON was already closed and should remain closed.
- W20 was allowed to edit only `benchmarks/**` and regenerated smoke artifacts.

## Goal And Expected Result
Perform read-only review and explicitly state:
- whether label leakage through acquisition-visible `sample_id`, `group_id`, `meta`, or schema is closed;
- whether strict JSON remains valid;
- whether any in-scope blocking defects, risks, questions, or required improvement requests remain.

## Responsibility Boundaries
Read-only. Do not edit files.

## In Scope
- Inspect all synthetic dataset builders for label-bearing acquisition-visible IDs/groups.
- Inspect `InMemoryBenchmarkProvider`.
- Inspect smoke `selections.csv` IDs/groups and JSON artifacts.
- Run cheap validation if useful.

## Out Of Scope
- SDK algorithm changes.
- New datasets.
- Timing/performance optional follow-ups.
- Full `ActiveLearningProject.run(...)` integration until SDK supports initial seed import.

## Files Or Modules May Be Changed
None.

## Files Or Areas Must Not Be Touched
Entire repository.

## Important Architectural Constraints And Forbidden Actions
- Do not require labels to be absent from synthetic text: class-informative text is the point of supervised classification.
- Do not require labels to be absent from post-selection diagnostic artifacts; diagnostics can use private labels after acquisition.
- Keep optional improvements separate from blocking findings.

## High-Level Execution Plan
- Read `benchmarks/sdk_first_benchmark.py`.
- Inspect generated smoke artifacts.
- Verify no label strings are visible in acquisition provider fields.
- Verify JSON strictness remains.
- Report findings and final gate status.

## Acceptance Criteria
- R29 P1 closed or concrete remaining leakage identified.
- No strict JSON regression.
- Final statement on benchmark harness readiness.

## Expected Tests And Validations
Optional:
- `python -m py_compile benchmarks/sdk_first_benchmark.py`
- `python benchmarks/sdk_first_benchmark.py --help`
- strict JSON parse/no NaN check
- provider leakage scan over all synthetic datasets

## Dependencies
Depends on W20.

## Parallel Or Sequential Notes
Sequential gate before SDK optimization.
