# W100 Reference Benchmark Acceptance Audit

## Context
The user requested benchmarks showing where this SDK is worse than analogs or direct manual work, with optional comparison to modAL and skactiveml and without polluting the repo. Existing benchmark code compares SDK selection against direct formulas, but external adapters may only be import checks.

## Goal
Create or improve a controlled benchmark artifact that compares SDK acquisition overhead/quality with manual direct formulas and, when available, real external-library adapters. The benchmark must be reproducible, local, and explicit about caveats.

## Responsibility Boundaries
Owned write scope:
- `benchmarks/acceptance_reference_comparison_2026_04_27.py`
- `benchmarks/results/acceptance_reference_2026_04_27/**`
- optional notes in `.agents/tmp/w100-benchmark-findings.md`

Read scope:
- `benchmarks/audit_sdk_vs_manual.py`
- `benchmarks/reference_strategy_benchmark.py`
- `tests/test_audit_benchmark_comparison.py`
- `docs/reference_active_learning_libraries.md`
- SDK strategy and scheduler code needed for benchmark wiring

Do not modify SDK production code, existing benchmark files, docs, pyproject, lock files, or tests.

## In Scope
- Manual direct uncertainty formula baseline.
- SDK StrategyScheduler path on identical probability fixtures.
- Import/status checks for modAL/skactiveml.
- Real external-library adapter only if already available/importable or trivial without dependency changes.
- CSV/JSON/Markdown benchmark results in the owned results directory.

## Out of Scope
- Installing new dependencies unless already present via the environment.
- Downloading datasets.
- Long-running benchmarks.
- Fixing SDK defects.

## Constraints
- Do not pollute global environment or lock files.
- Keep results small and deterministic.
- Set `PYTHONDONTWRITEBYTECODE=1` when running benchmark commands if possible.
- Explicitly label caveats and unsupported external comparisons.

## Execution Plan
1. Inspect existing benchmark harnesses.
2. Reuse existing benchmark logic where clean; do not duplicate large code unnecessarily.
3. Add the owned benchmark script and generate one small result set.
4. Validate artifact schemas by reading generated CSV/JSON/Markdown.
5. Record benchmark conclusions and caveats in `.agents/tmp/w100-benchmark-findings.md`.

## Acceptance Criteria
- Benchmark script exists and is runnable.
- Results directory contains `comparison.csv`, `summary.json`, and `analysis.md`.
- Output explicitly states whether SDK is worse than manual or external references in measured dimensions.
- No dependency or lock-file pollution.

## Dependencies
Can run in parallel with runtime and strategy subtasks. No write-scope overlap.
