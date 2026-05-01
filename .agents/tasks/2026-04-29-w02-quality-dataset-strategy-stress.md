# 2026-04-29 W02 Quality Dataset Strategy Stress

## Task Identifier

W02 - Quality, strategy, and dataset stress.

## Context

Part of BLACKBOX-STRESS-MASTER. The SDK documentation claims diagnostic active-learning quality evidence and lists many strategies/schedulers. This task attacks metric weakness and strategy collapse using black-box usage.

## Goal

Find low metrics, strategy regressions versus random, selection collapse, excessive duplicate/group concentration, poor rare-class coverage, calibration problems, and runtime scaling issues across multiple datasets/settings.

## Responsibility Boundaries

Write scope:
- `.agents/tmp/blackbox_stress_2026_04_29/w02_quality/**`

Readable sources:
- `README.md`
- `docs/README.md`
- `docs/BENCHMARK_EVIDENCE.md`
- `benchmarks/README.md`
- W02-generated artifacts

Must not read:
- `src/active_learning_sdk/**`
- `tests/**`
- benchmark implementation source

## In Scope

- Use documented benchmark commands when useful.
- Create independent black-box benchmark scripts if faster.
- Synthetic datasets: separable, rare-class trap, grouped duplicates, noisy labels, adversarial duplicates, label imbalance.
- Optional capped real datasets from documentation if runtime permits.
- Strategies documented in README, including random, uncertainty, group/class balanced, embedding/diversity, BADGE, adaptive, stochastic/committee where a black-box adapter can provide capabilities.

## Out Of Scope

- Scientific claim of universal superiority/inferiority.
- Modifying benchmark harness source.
- Reading benchmark source for implementation details.

## Execution Plan

1. Run a fast documented smoke/quality benchmark if dependencies are available.
2. Run independent black-box dataset/strategy sweeps with bounded sample counts.
3. Compare every non-random strategy to matched random by seed/dataset/budget.
4. Flag low macro-F1/AULC, negative lift, selection collapse, excessive concentration, and slow paths.
5. Produce `findings.md`, `metrics.csv`, and `summary.json`.

## Acceptance Criteria

- At least 3 datasets or dataset variants.
- At least 5 strategies or scheduler configurations.
- At least 2 seeds for synthetic stress.
- Findings distinguish weak evidence from actual SDK failure.

## Parallelism

Can run in parallel with W01, W03, W04.
