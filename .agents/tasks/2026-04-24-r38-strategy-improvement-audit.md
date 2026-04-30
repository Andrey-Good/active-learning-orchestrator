# R38 - Strategy Improvement Audit

## Relation To Overall Task
The user asked to improve existing heuristics and methods scientifically. After accepting the benchmark harness, we need a focused read-only audit of current strategy/scheduler implementation to identify high-value changes with measurable benchmark hypotheses.

## Assumptions And Resolved Ambiguities
- Do not edit files in this audit.
- Existing accepted changes include import-labels and seed train before first select.
- Benchmark datasets include redundancy and rare-class regimes.

## Goal And Expected Result
Produce a ranked list of SDK acquisition/scheduler improvements that can be implemented surgically and tested with the new benchmarks. For each candidate include:
- current code location;
- likely failure mode;
- expected metric improvement;
- benchmark slice to prove or falsify it;
- risk/complexity.

## Responsibility Boundaries
Read-only audit. Do not edit files.

## In Scope
- `src/active_learning_sdk/strategies/**`
- `src/active_learning_sdk/engine.py` scheduler and selection context.
- Config surface needed for strategies.
- Benchmark signals needed to choose first implementation.

## Out Of Scope
- Writing code.
- Running long benchmarks.
- External research/web search unless absolutely necessary.

## Files Or Modules May Be Changed
None.

## Files Or Areas Must Not Be Touched
Entire repository.

## Important Architectural Constraints And Forbidden Actions
- Do not propose huge rewrites as first step if a surgical improvement can be measured.
- Do not treat random underperformance/overperformance as proof by itself; tie to a failure mode.
- Prioritize changes testable by current benchmark harness.

## High-Level Execution Plan
- Inspect strategy and scheduler code.
- Use benchmark design context to map failure modes.
- Return ranked candidates and suggested first worker task.

## Acceptance Criteria
- Clear recommendation for next implementation task.
- Each candidate has a metric and benchmark slice.

## Expected Tests And Validations
Read-only; no tests required.

## Dependencies
Can run in parallel with W28 baseline run.

## Parallel Or Sequential Notes
Parallel with W28.
