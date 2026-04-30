# Optimization Cycle Plan - Existing Active Learning Behavior

## Objective
Improve existing active-learning quality levers using the accepted benchmark evidence:
- baseline sweep;
- acquisition diagnostics;
- warm-start rejection.

The goal is not to add large missing methods yet. The goal is to improve correctness, reproducibility, and measurable strategy quality for current uncertainty/random/mix behavior.

## Current Evidence
- `bert_tiny` is not a reliable quality baseline.
- `tiny_transformer_unfrozen_default` passes random@48 but has high variance and near-saturates by 80 labels.
- `tiny_transformer_unfrozen_fixed_dataset_default` is more stable and useful for acquisition effects, but misses random@48 macro-F1 threshold.
- Uncertainty heuristics often select class-skewed batches.
- Warm-start reduces skew but fails strict quality criteria.
- Existing SDK strategies lack probability validation and deterministic seeded tie-breaking.
- `mix` mode can distort weights because each strategy selects from the full pool and duplicates are deduped later.

## Hypotheses
1. Correctness hypothesis: probability validation and deterministic tie-breaking reduce silent bad selections and make SDK behavior reproducible.
2. Scheduler hypothesis: mix mode with exclusion-aware allocation better preserves requested exploration/exploitation weights.
3. Quality hypothesis: predicted-class-balanced uncertainty can reduce class skew and improve AULC/final macro-F1 compared with pure uncertainty and random on fixed-dataset controls.
4. Calibration/smoothing hypothesis: uncertainty score smoothing or temperature-based acquisition may reduce overconfident class collapse.

## Decomposition
- W12: SDK strategy correctness, probability validation, deterministic tie-break, tests.
- R19: research candidate benchmark improvements using accepted diagnostics.
- W13: benchmark-only quality experiment for balanced/smoothed uncertainty candidates.
- Review tasks after every worker change.
- Follow-up worker only promotes benchmark-proven changes into SDK public strategies.

## Ownership Boundaries
- SDK correctness worker owns:
  - `src/active_learning_sdk/strategies/uncertainty.py`
  - relevant tests.
- Benchmark experiment worker owns:
  - `benchmarks/run_learning_curve_experiments.py`
  - `benchmarks/results/learning_curves/*balanced*` or similar artifacts.
- Scheduler worker, if launched later, owns:
  - `src/active_learning_sdk/engine.py`
  - relevant tests.

No two workers should edit the same files in parallel.

## Parallel/Sequential Rules
- R19 can run in parallel with W12.
- W13 must wait for R19 recommendations unless narrowly scoped.
- Promotion into SDK must wait for W13 benchmark review.

## Completion Criteria
- Tests pass.
- Benchmark artifacts show whether each hypothesis is supported or rejected.
- Reviewers explicitly clear worker outputs.
- Final report includes before/after metrics and remaining risks.
