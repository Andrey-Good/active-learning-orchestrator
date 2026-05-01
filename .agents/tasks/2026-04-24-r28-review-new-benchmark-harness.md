# R28 - Review New SDK-First Benchmark Harness

## Relation To Overall Task
W18 created a replacement benchmark harness after the old notebooks/tests/benchmark layer was removed. This review checks whether the new benchmark layer is correct enough to drive later SDK algorithm improvements.

## Assumptions And Resolved Ambiguities
- Old notebooks/tests/benchmark code were intentionally removed before W18.
- W18 was allowed to create `benchmarks/**` and small smoke artifacts.
- W18 was not allowed to change `src/active_learning_sdk/**`, README, docs, or docker.
- Existing dirty changes outside `benchmarks/**` and W18 task docs may predate W18. Do not infer W18 touched unrelated files from `git status`; use file contents and W18 ownership boundaries.
- The harness is allowed to use SDK `StrategyScheduler`/`SelectionContext` rather than full `ActiveLearningProject` because the current project API lacks public initial-seed import. Treat this as a known SDK gap, not automatically a defect.

## Goal And Expected Result
Perform an independent read-only review of:
- `benchmarks/README.md`
- `benchmarks/sdk_first_benchmark.py`
- smoke artifacts under `benchmarks/results/smoke/`

Report concrete defects, risks, required improvement requests, optional follow-ups, and questions. If satisfied, explicitly state there are no remaining in-scope defects, blocking risks, blocking questions, or required improvement requests.

## Responsibility Boundaries
Read-only review. Do not edit files.

## In Scope
- Correctness of benchmark loop and metrics.
- Whether SDK strategies/scheduler are really used for acquisition.
- Whether random baseline/lift calculations are fair.
- Dataset leakage or accidental use of true labels beyond the documented initial-seed policy.
- Determinism and seed handling.
- Artifact schema usefulness and readability.
- Whether smoke validation proves the harness runs.
- Methodological risks that would make later algorithm conclusions invalid.

## Out Of Scope
- SDK algorithm implementation changes.
- README product rewrite.
- Dependency cleanup in `pyproject.toml`, except note if benchmark docs depend on missing dependencies.
- Complaints about pre-existing dirty files outside W18 scope.

## Files Or Modules May Be Changed
None.

## Files Or Areas Must Not Be Touched
Entire repository. This is read-only.

## Important Architectural Constraints And Forbidden Actions
- Do not propose reintroducing notebooks.
- Do not require `ActiveLearningProject.run(...)` in this review unless you can point to a clean public initial-seed path that W18 missed.
- Do not count “smoke datasets are easy” as a blocking defect unless the harness cannot run a broader suite or falsely claims smoke is final proof.

## High-Level Execution Plan
- Read `benchmarks/README.md`.
- Read `benchmarks/sdk_first_benchmark.py`.
- Inspect smoke `manifest.json`, `metrics.csv`, `selections.csv`, `summary.json`, `summary.md`.
- Optionally rerun `python benchmarks/sdk_first_benchmark.py --help` and a tiny smoke if cheap.
- Produce review findings with evidence and severity.

## Acceptance Criteria
- Review identifies any correctness/methodology defects that must be fixed before SDK optimization begins.
- Review separates blocking required fixes from optional improvement ideas.
- If no blocking issues remain, review explicitly says so.

## Expected Tests And Validations
Read-only inspection. Optional:
- `python benchmarks/sdk_first_benchmark.py --help`
- `python -m py_compile benchmarks/sdk_first_benchmark.py`

## Dependencies
Depends on W18 completion.

## Parallel Or Sequential Notes
Sequential gate before treating the benchmark harness as accepted.
