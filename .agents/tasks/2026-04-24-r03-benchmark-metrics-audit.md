# Task ID: 2026-04-24-r03-benchmark-metrics-audit

## Relation To Overall Task

This is the benchmark and metric research slice. It designs how to measure whether SDK changes improve active learning quality.

## Goal

Audit existing notebooks and benchmark runner, then propose a better experimental framework for iterative hypothesis testing.

## Responsibility Boundaries

Read-only. Do not edit files.

## In Scope

- `active_learning_lab.ipynb`
- `lab/active_learning_lab.ipynb`
- `benchmarks/run_extended_benchmarks.py`
- `benchmarks/results/*.csv`
- current benchmark summary artifacts

## Out Of Scope

- implementing a new benchmark runner;
- modifying notebooks;
- changing results.

## Files That May Be Changed

None.

## Files That Must Not Be Touched

The entire repository. Review only.

## Architectural Constraints

- Active learning must be evaluated over label budgets, not only final metrics.
- Random baseline must be strong enough to verify learning.
- Multiple seeds are required for serious claims.
- Metrics should separate model quality, label efficiency, runtime, and SDK overhead.

## Execution Plan

1. Inspect existing benchmark matrix and artifacts.
2. Identify missing metrics and misleading metrics.
3. Propose a notebook/script template for controlled experiments.
4. Propose baseline datasets/models.
5. Propose artifact schemas.

## Acceptance Criteria

- Output includes recommended metrics.
- Output includes recommended experiment design.
- Output identifies the smallest useful next notebook/runner to build.
