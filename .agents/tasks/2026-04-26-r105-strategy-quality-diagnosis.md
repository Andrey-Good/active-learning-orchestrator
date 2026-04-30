# Task r105: Strategy Quality Diagnosis

## Context
The user wants the active-learning SDK to improve benchmark quality materially, not merely run correctly. Recent real-data probes show most heuristics underperform random, especially entropy on Banking77. This task is a read-only research diagnosis for the strategy implementations.

## Goal
Identify algorithmic or implementation reasons why built-in strategies underperform random and propose concrete SDK changes likely to improve macro-F1, class coverage, and zero-recall metrics under small label budgets.

## Responsibility Boundaries
You own read-only analysis of `src/active_learning_sdk/strategies/*`, `src/active_learning_sdk/engine.py`, and related tests.

## In Scope
- Inspect uncertainty, embedding, BADGE, stochastic/committee, hybrid, and scheduler mix logic.
- Look for cold-start, many-class, class-coverage, diversity, tie-breaking, normalization, and batch-selection issues.
- Propose specific changes with expected metric impact and risks.

## Out of Scope
- Do not edit files.
- Do not run long external dataset jobs.
- Do not change benchmark code.

## Files May Be Read
- `src/active_learning_sdk/strategies/*`
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/configs.py`
- `tests/test_*strategy*.py`

## Files Must Not Be Touched
- All files. This is read-only.

## Constraints
- Preserve deterministic behavior.
- Do not recommend leaking oracle labels into selection.
- Prefer product-grade guardrails over benchmark-specific hacks.

## Plan
1. Inspect how strategies rank and batch-select samples.
2. Identify likely causes of entropy/BADGE/diversity underperformance.
3. Propose a small set of high-impact SDK improvements.

## Acceptance Criteria
- Final answer lists concrete causes and candidate fixes.
- Each fix includes likely metric impact and risk.
- Notes whether the change belongs in SDK strategy logic or only benchmark adapter/harness.

## Dependencies
None. Can run in parallel with benchmark-adapter diagnosis.
