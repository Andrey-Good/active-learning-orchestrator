# Task r108: BADGE Quality Diagnosis

## Context
The user asked to solve the BADGE quality problem. Current real probes show BADGE underperforms the new cold-start-guarded uncertainty strategies and is slow on Banking77. Prior diagnosis found many-class cold-start class-discovery failure: model probabilities and gradient embeddings only meaningfully cover labels already observed in seed data.

## Goal
Read-only diagnosis of `BadgeStrategy` and benchmark gradient embeddings. Identify concrete SDK changes that improve BADGE quality and runtime without oracle-label leakage.

## Responsibility Boundaries
Read-only analysis. Do not edit files.

## In Scope
- `src/active_learning_sdk/strategies/badge.py`
- `src/active_learning_sdk/strategies/embedding.py`
- `src/active_learning_sdk/strategies/uncertainty.py`
- `benchmarks/sdk_first_benchmark.py` BADGE/gradient embedding code
- BADGE tests

## Out of Scope
- Do not change benchmark labels or use oracle labels in acquisition.
- Do not tune on test labels inside the strategy.
- Do not edit files.

## Questions
- Why does BADGE underperform on Banking77 budget=100?
- Is the issue algorithmic, adapter-proxy related, or cold-start related?
- What product-grade fallback/guardrail should BADGE use in sparse label-support regimes?
- Are there safe normalization/projection/runtime changes to make BADGE faster?

## Acceptance Criteria
- Final answer lists concrete fixes with risks.
- Explicitly states whether fixes belong in SDK, benchmark adapter, or both.

## Dependencies
None. Can run in parallel with local experiments.
