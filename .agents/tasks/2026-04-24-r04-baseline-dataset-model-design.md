# Task ID: 2026-04-24-r04-baseline-dataset-model-design

## Relation To Overall Task

This slice designs a baseline dataset/model pair where even random selection learns reliably under a small label budget.

## Goal

Identify and specify a baseline experimental setup for active learning optimization.

## Responsibility Boundaries

Read-only. Do not edit files.

## In Scope

- existing synthetic dataset generation in notebooks;
- current model registry;
- current benchmark results;
- candidate lightweight datasets already used or cached;
- simple baseline model options already available in dependencies.

## Out Of Scope

- downloading large new models;
- implementing new model adapters;
- changing notebooks;
- adding dependencies.

## Files That May Be Changed

None.

## Files That Must Not Be Touched

The entire repository. Review only.

## Architectural Constraints

- Baseline must train quickly.
- Random must show a positive learning curve.
- Setup must be small enough for repeated hypothesis cycles.
- It should expose active learning differences without being too easy.

## Execution Plan

1. Inspect current synthetic dataset and benchmark results.
2. Inspect model options and dependencies.
3. Propose baseline dataset/model/budget setup.
4. Define success metrics for "random learns normally".
5. Suggest variants with controlled difficulty.

## Acceptance Criteria

- Output gives one primary baseline recommendation and alternatives.
- Output includes budgets, expected metrics, and why the setup is useful.
- Output warns about setups that are too easy or too noisy.
