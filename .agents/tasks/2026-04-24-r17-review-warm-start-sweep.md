# R17 - Review Warm-Start Strategy Sweep

## Relation to Overall Task
Independent review of W11 warm-start optimization experiment. This determines whether the first tested improvement hypothesis is accepted or rejected.

## Assumptions and Resolved Ambiguities
- W11 should not have changed SDK source or README.
- Warm-start schedules are benchmark-only composites, not new SDK strategies.
- Claimed result: warm-start reduces skew but does not improve quality enough versus random and pure uncertainty baselines.

## Goal
Validate warm-start runner changes, artifacts, summary math, and conclusions.

## Responsibility Boundaries
- Read-only review.
- Do not edit files.

## In Scope
- Review `benchmarks/run_learning_curve_experiments.py` warm-start code.
- Review `benchmarks/results/learning_curves/warm_start_*`.
- Recompute:
  - row counts and unique keys;
  - AULC/final metrics;
  - lift vs random and pure baselines;
  - skew deltas;
  - pass/fail of strict warm-start criterion.
- Verify existing sweep/diagnostics behavior was not broken at least by code inspection or short commands.

## Out of Scope
- Do not implement fixes.
- Do not propose new methods as blocking.
- Do not rerun full training unless necessary; artifact recompute is preferred.

## Files/Areas May Read
- `benchmarks/run_learning_curve_experiments.py`
- `benchmarks/results/learning_curves/warm_start_*`
- Related baseline/diagnostics artifacts as needed.

## Files/Areas Must Not Touch
- Entire repo is read-only.

## Architectural Constraints and Forbidden Actions
- Negative experiment results are acceptable if measured correctly.
- Block for misleading math, wrong baselines, artifact key collisions, or hidden behavior changes.

## Execution Plan
- Inspect code -> verify composite schedule semantics and no SDK changes.
- Inspect artifacts -> verify schemas/row counts/statuses.
- Recompute key metrics -> compare to summary claims.
- Report findings.

## Acceptance Criteria
- If clean, explicitly state no remaining in-scope defects, blocking risks, blocking questions, or required improvement requests.
- If not clean, list concrete blocking findings and expected fixes.

## Expected Tests and Validations
- Artifact recompute script.
- Optional `py_compile`.
- Optional command smoke if feasible.

## Dependencies
- Depends on W11.

## Parallel/Sequential Notes
- Must complete before warm-start conclusion is used in final report.
