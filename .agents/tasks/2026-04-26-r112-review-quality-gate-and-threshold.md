# Task r112: Review Quality Gate and Threshold Changes

## Context
The orchestrator added a quality-gate report CLI, changed the uncertainty cold-start support threshold from 0.5 to 0.95, and retained BADGE cold-start/runtime fixes.

## Goal
Review whether these changes are correct, non-leaky, deterministic, and sufficiently tested.

## Responsibility Boundaries
Read-only review. Do not edit files.

## In Scope
- `benchmarks/quality_gate_report.py`
- `tests/test_quality_gate_report.py`
- `src/active_learning_sdk/strategies/uncertainty.py`
- `src/active_learning_sdk/strategies/badge.py`
- `benchmarks/sdk_first_benchmark.py`
- `tests/test_badge_strategy.py`
- `tests/test_class_balanced_entropy_strategy.py`
- `tests/test_sdk_first_benchmark_embedding_diagnostics.py`

## Out of Scope
- Do not review unrelated dirty worktree changes.
- Do not propose oracle-label leakage.

## Review Questions
- Are quality-gate calculations correct enough for release diagnostics?
- Are gates too weak/misleading for product-quality claims?
- Is uncertainty threshold 0.95 defensible and safe?
- Are BADGE runtime changes deterministic and bounded?
- Are tests adequate?

## Acceptance Criteria
- Final answer lists findings with severity and file/line references.
- If no blocking findings, state residual risks.

## Dependencies
Depends on current local patch.
