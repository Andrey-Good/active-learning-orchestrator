# R89 - Final Review Stage 6 Stop Criteria And Benchmarks

## Context

Stage 6 added:

- SDK stop criteria with min gates, max limits, metric plateau, acquisition score convergence, label distribution stabilization, calibration stabilization, effective batch clipping, and persistent stop traces.
- Robust handling of malformed stop metric values.
- Benchmark stop-policy simulation that compares auto-stop policies against fixed final budgets.

Prior review found malformed calibration metrics could crash stop evaluation. W71 fixed this. W70 added benchmark artifact wiring.

## Goal

Perform a final read-only review of Stage 6 before the orchestrator moves to Stage 7.

## Responsibility Boundaries

You are a reviewer. Do not edit files.

## In Scope

- `src/active_learning_sdk/configs.py`
- `src/active_learning_sdk/engine.py`
- `tests/test_stop_criteria.py`
- `benchmarks/sdk_first_benchmark.py`
- `tests/test_sdk_first_benchmark_embedding_diagnostics.py`
- `benchmarks/README.md`

## Out Of Scope

- Stage 7 label backend work.
- Stage 8 reporting work.
- Stage 9 README release polish.
- New strategy algorithms.

## Required Review Questions

- Are stop criteria safe on edge cases: empty pools, max budgets, min gates, sparse acquisition traces, insufficient label distribution rounds, missing/malformed metrics?
- Do stop traces persist and explain both stop and non-stop decisions?
- Does benchmark stop simulation avoid using test-label metrics during acquisition, i.e. is it clearly post-hoc analysis of generated curves?
- Do stop-policy rows contain enough information to compare label savings, runtime savings, and quality delta versus fixed full budget?
- Are CSV/JSON outputs strict-serializable and documented?
- Are tests adequate for the new behavior without making long benchmark runs part of the suite?

## Explicitly Forbidden

- Do not edit files.
- Do not broaden the scope beyond Stage 6.
- Do not revert unrelated repository changes.

## Validation To Run

- `uv run --group dev pytest -q tests/test_stop_criteria.py`
- `uv run --group dev pytest -q tests/test_sdk_first_benchmark_embedding_diagnostics.py`
- `uv run --group dev pytest -q`
- Optional temp smoke if time permits:
  `uv run python benchmarks/sdk_first_benchmark.py --preset smoke --datasets grouped_duplicates --strategies random,entropy --budgets 12,18 --seeds 13 --output-dir <temp>`

## Output

Return severity-ordered findings with file/line references. If no findings remain, say so explicitly and include validation results.
