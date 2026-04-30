# Task r106: Benchmark Adapter Diagnosis

## Context
The benchmark uses `SklearnTextBenchmarkAdapter` and real Hugging Face datasets. Current real probes show many strategies worse than random. The SDK may be fine while the benchmark adapter gives poor signals, or the benchmark may expose real SDK weaknesses.

## Goal
Diagnose whether benchmark/model adapter choices suppress active-learning signal and propose safe changes that make benchmarks better reflect real SDK quality without cheating.

## Responsibility Boundaries
You own read-only analysis of `benchmarks/sdk_first_benchmark.py`, benchmark results, and adapter behavior inside that file.

## In Scope
- Inspect `SklearnTextBenchmarkAdapter` probabilities, embeddings, gradient embeddings, stochastic proxies, committee proxies, initial seed, and curve loop.
- Check whether embeddings/gradient embeddings are too sparse, unnormalized, or stale.
- Check whether real dataset caps and budgets make comparisons noisy.
- Propose benchmark/adapter improvements that are fair and product-relevant.

## Out of Scope
- Do not edit files.
- Do not add datasets.
- Do not run long benchmark sweeps.

## Files May Be Read
- `benchmarks/sdk_first_benchmark.py`
- `benchmarks/results/runtime/*`
- `tests/test_sdk_first_benchmark*.py`

## Files Must Not Be Touched
- All files. This is read-only.

## Constraints
- No oracle-label leakage into selection.
- Preserve deterministic reproducibility.
- Keep smoke benchmarks under a few minutes.

## Plan
1. Inspect adapter training/proba/embedding/gradient methods.
2. Compare with reported metrics and selection diagnostics.
3. Recommend fair modifications likely to improve strategy signal.

## Acceptance Criteria
- Final answer names benchmark artifacts/adapter behaviors that likely hurt metrics.
- Proposes concrete changes and expected effects.
- Flags anything that would be benchmark cheating.

## Dependencies
None. Can run in parallel with strategy-quality diagnosis.
