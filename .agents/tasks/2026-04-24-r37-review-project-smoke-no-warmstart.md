# R37 - Review Project Smoke Without Warm-Start

## Relation To Overall Task
W27 removed the project-smoke external model warm-start after the SDK seed-train feature was accepted. This review checks that the artifact now proves public SDK seed train behavior.

## Assumptions And Resolved Ambiguities
- W27 was allowed to edit `benchmarks/**` only.
- Expected first project smoke step is seed `TRAIN_EVAL` with `round_id=None`.
- Expected later active sequence is `SELECT/PUSH/WAIT/PULL/TRAIN_EVAL/UPDATE`.

## Goal And Expected Result
Read-only review of W27. Explicitly state whether project smoke is accepted without warm-start and whether any in-scope blocking issues remain.

## Responsibility Boundaries
Read-only. Do not edit files.

## In Scope
- `benchmarks/sdk_first_benchmark.py` project smoke path.
- `benchmarks/README.md`.
- `benchmarks/results/project_smoke/**`.
- Verify no direct `model.fit(...)` before first SDK `run_step`.
- Verify strict JSON and artifact claims.

## Out Of Scope
- SDK code changes.
- Full strategy matrix.
- Root README rewrite.

## Files Or Modules May Be Changed
None.

## Files Or Areas Must Not Be Touched
Entire repository.

## High-Level Execution Plan
- Inspect code/artifacts.
- Optionally run strict JSON parse.
- Report findings and acceptance status.

## Acceptance Criteria
- No private state mutation or external warm-start.
- Public seed train step is recorded.
- Active round completes.

## Expected Tests And Validations
Optional:
- strict JSON parse of `project_smoke/summary.json`

## Dependencies
Depends on W27.

## Parallel Or Sequential Notes
Sequential before moving to strategy-quality tasks.
