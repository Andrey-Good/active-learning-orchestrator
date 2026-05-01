# W28 - Run Current Honest Baseline Matrix

## Relation To Overall Task
The benchmark harness and project smoke are accepted. Before changing SDK acquisition algorithms, we need fresh baseline numbers from the new honest benchmark layer.

## Assumptions And Resolved Ambiguities
- Use the accepted benchmark harness under `benchmarks/`.
- Do not change SDK code.
- It is acceptable to generate benchmark artifacts under a new result directory.
- The goal is evidence for next SDK improvements, not final publication-scale benchmarking.

## Goal And Expected Result
Run a baseline matrix that includes all current synthetic datasets and strategies, with multiple seeds where runtime allows. Produce a compact report identifying:
- which strategies beat random by dataset and budget;
- where uncertainty fails;
- where redundancy/group concentration suggests diversity is needed;
- which metric should drive the next SDK improvement.

## Responsibility Boundaries
Owned by this worker:
- Generate new artifacts under `benchmarks/results/baseline_current/**`.
- Optionally add a short markdown report under that same directory.

Do not change:
- `src/**`
- benchmark source code unless a run-blocking bug is found, in which case report it instead of editing.
- tests/docs/root README.

## In Scope
- Run `benchmarks/sdk_first_benchmark.py` using full/current matrix or an equivalent explicit matrix.
- Include `separable_topics`, `rare_class_trap`, and `grouped_duplicates`.
- Include strategies: random, entropy, margin, least_confidence, mix_entropy_random, mix_uncertainty_random.
- Use at least three seeds if runtime remains reasonable.
- Summarize macro-F1 AULC, lift vs random, rare recall, group HHI/top-group concentration, and runtime.

## Out Of Scope
- SDK implementation changes.
- Benchmark harness redesign.
- External datasets/models.

## Files Or Modules May Be Changed
- `benchmarks/results/baseline_current/**`

## Files Or Areas Must Not Be Touched
- `src/**`
- `tests/**`
- `benchmarks/sdk_first_benchmark.py`
- root README/docs/docker

## Important Architectural Constraints And Forbidden Actions
- Do not edit algorithm code.
- Do not hide negative results.
- If a run fails, capture command, failure, and partial artifacts.

## High-Level Execution Plan
- Run baseline benchmark with full preset or explicit args.
- Inspect outputs and write `analysis.md` in the result directory.
- Include concrete recommended next improvement target.

## Acceptance Criteria
- Artifacts exist and are strict JSON/CSV-readable.
- Report names at least one measurable failure mode and one next hypothesis.

## Expected Tests And Validations
- Benchmark command(s) run successfully.
- Strict JSON parse for generated JSON artifacts.
- CSV row count checks.

## Dependencies
Depends on accepted benchmark harness and project smoke.

## Parallel Or Sequential Notes
Can run in parallel with read-only strategy audit.
