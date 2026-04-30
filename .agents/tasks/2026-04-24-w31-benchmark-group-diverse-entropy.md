# W31 - Benchmark Group-Diverse Entropy

## Relation To Overall Task
The SDK now includes accepted `group_diverse_entropy`. We need to add it to the benchmark harness and test whether it improves the measured failure mode: uncertainty over-concentration on `grouped_duplicates`, while preserving rare-class gains.

## Assumptions
- SDK implementation is accepted by R39/R40.
- Benchmark harness can be edited in this task.
- Generated artifacts should go under a new result directory.

## Goal
Register `group_diverse_entropy` in `benchmarks/sdk_first_benchmark.py` strategy configs, run a targeted benchmark, and write a concise conclusion.

## Responsibility Boundaries
Owned by this worker:
- `benchmarks/sdk_first_benchmark.py`
- `benchmarks/README.md` if needed
- `benchmarks/results/group_diverse_entropy/**`

Do not change:
- `src/**`
- `tests/**`
- root README/docs/docker

## In Scope
- Add benchmark strategy spec for `group_diverse_entropy`.
- Run targeted benchmark:
  - datasets: `grouped_duplicates`, `rare_class_trap`
  - strategies: `random`, `entropy`, `group_diverse_entropy`, `mix_uncertainty_random`
  - seeds: `13,17,23`
  - budgets: `16,32,48,64,96`
- Compare macro-F1 AULC, early budget macro-F1, group HHI/top-group fraction, rare recall.
- Write `analysis.md` with hypothesis result: confirmed/rejected/mixed.

## Out Of Scope
- SDK changes.
- Full matrix with every strategy unless cheap and useful.
- Claiming final product superiority.

## Files Or Modules May Be Changed
- `benchmarks/sdk_first_benchmark.py`
- `benchmarks/README.md`
- `benchmarks/results/group_diverse_entropy/**`

## Files Or Areas Must Not Be Touched
- `src/**`
- `tests/**`
- root `README.md`
- `docs/**`
- `docker/**`

## Acceptance Criteria
- Strategy appears in benchmark CLI and artifacts.
- Targeted artifacts exist and parse.
- Analysis states whether group concentration improved and whether quality improved/regressed.

## Expected Validations
- `python -m py_compile benchmarks/sdk_first_benchmark.py`
- Targeted benchmark command
- Strict JSON parse and CSV row count checks

## Dependencies
Depends on R40.

## Parallel Or Sequential Notes
Sequential before deciding next algorithm iteration.
