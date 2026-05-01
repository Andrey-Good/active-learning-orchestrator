# r114 - Review Blocker Fixes

## Context

Three blocker fixes were implemented after final product-quality review:

- strengthened quality gate;
- benchmark validity fixes for capped labels and acquisition-time diagnostics;
- `strict_capabilities` contract support.

## Goal

Review whether these fixes actually address the reported blockers without introducing regressions or misleading product claims.

## Responsibility Boundaries

Read-only review. Do not edit files.

## In Scope

- `benchmarks/quality_gate_report.py`
- `tests/test_quality_gate_report.py`
- `benchmarks/sdk_first_benchmark.py`
- `tests/test_sdk_first_benchmark_embedding_diagnostics.py`
- `src/active_learning_sdk/engine.py`
- `tests/test_strategy_capabilities.py`
- `README.md` benchmark/status claims

## Out Of Scope

- No implementation changes.
- No long benchmark runs.
- No dependency changes.

## Review Questions

1. Does the quality gate now reject random-equivalent ties?
2. Are capped real-dataset labels computed only from retained records?
3. Are selected embedding diagnostics measured from acquisition-time model state?
4. Does `strict_capabilities=False` allow configuration/attachment while still failing clearly at runtime if the strategy is used without required capability?
5. Does README accurately describe the current evidence, especially Banking77 budget-300 failure vs budget-500 pass?

## Acceptance Criteria

- Findings ordered by severity with file/line references.
- Explicit verdict on whether the original blocker findings are resolved.
- Call out any remaining blocker to a controlled quality-product claim.

## Dependencies

Depends on w88, w89, and w90.
