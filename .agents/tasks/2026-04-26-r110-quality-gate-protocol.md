# Task r110: Quality Gate Protocol

## Context
The user wants enough benchmark evidence to honestly say whether the SDK is product-quality. Single-seed/single-budget metrics are insufficient.

## Goal
Design a product-quality active-learning benchmark protocol and acceptance gates.

## Responsibility Boundaries
Read-only analysis. Do not edit files.

## In Scope
- Existing `benchmarks/sdk_first_benchmark.py`
- Existing benchmark outputs
- Metrics needed for active learning quality claims

## Out of Scope
- Do not run long benchmarks.
- Do not edit code.

## Questions
- Which datasets/budgets/seeds should be used for a practical local quality gate?
- Which strategies should be compared?
- What pass/fail criteria are defensible?
- What claims are allowed if criteria pass?

## Acceptance Criteria
- Final answer gives concrete protocol and thresholds.
- Distinguish internal smoke gate vs publishable evidence.

## Dependencies
None.
