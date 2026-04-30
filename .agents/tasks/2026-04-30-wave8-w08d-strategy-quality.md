# W08D Strategy And Quality Stress

Task identifier: `wave8-w08d-strategy-quality`

## Goal

Stress documented active-learning strategy behavior and benchmark quality claims through black-box runs, focusing on realistic failure modes: small budgets, rare classes, duplicate-heavy groups, many-class cold starts, strategy concentration, and regressions versus matched random.

## Ownership

May write only under `.agents/tmp/blackbox_stress_wave8/w08d_strategy_quality/`.

Must not touch product files, docs, tests, benchmark source, or other workers' directories.

## In Scope

- Public SDK usage with generated synthetic datasets and custom model adapters.
- Documented strategy and scheduler names from README.
- Documented benchmark CLIs may be executed; do not read benchmark implementation source.
- Metrics comparing strategy outcomes against matched random under identical data/model/budget/seed.
- Duplicate/group concentration diagnostics and zero-recall/coverage signals where externally observable.

## Out Of Scope

- Reading implementation source under `src/**`.
- Reading repository tests under `tests/**`.
- Reading benchmark implementation source under `benchmarks/*.py`.
- Treating a strategy losing on one hostile fixture as a correctness bug without a documented claim violation.

## Plan

1. Build small black-box synthetic fixtures with stable sample ids, labels, text, optional group ids, and deterministic model behavior.
2. Run a matrix of documented strategies and schedulers under matched seeds/budgets.
3. Compare against random and report product-quality warnings separately from correctness defects.
4. Execute a documented benchmark quality command if feasible and inspect only generated artifacts.
5. Write `findings.md`, `metrics.csv`, `summary.json`, and reproduction commands.

## Acceptance Criteria

- Confirmed correctness defects are separated from diagnostic quality limitations.
- Every quantitative claim has artifact evidence.
