# 2026-04-28-w03-strategy-correctness-audit

## Context

Part of a senior acceptance audit requested on 2026-04-28. The target is strategy correctness, mathematical soundness, edge-case handling, and consistency with active learning expectations.

## Goal

Stress all selection strategies and adapters for invalid probability shapes, NaN/Inf, ties, class imbalance, group constraints, budget exhaustion, embedding dimensionality, duplicate IDs, deterministic seeds, and capability mismatches. Produce focused tests plus findings.

## Responsibility Boundaries

Owner may change only:

- `tests/test_deep_audit_strategy_correctness_2026_04_28.py`
- `.agents/tmp/2026-04-28-w03-strategy-correctness-findings.md`

Owner must not change:

- `src/**`
- existing `tests/**`
- `benchmarks/**`
- docs other than the owned findings file

## In Scope

- `strategies/base.py`
- `strategies/uncertainty.py`
- `strategies/embedding.py`
- `strategies/badge.py`
- `strategies/stochastic.py`
- `strategies/adaptive.py`
- `strategies/hybrid.py`
- adapter capability contracts when needed for strategy inputs.

## Out of Scope

- Runtime persistence and backend orchestration.
- Packaging metadata.
- Benchmark framework implementation.

## Constraints

- Use small deterministic arrays and fake adapters/models.
- Tests must make mathematically defensible assertions.
- Avoid claims that are only subjective preference.

## Execution Plan

1. Read strategy APIs and existing strategy tests.
2. Identify untested edge cases and active-learning contract violations.
3. Add failing/regression tests for concrete correctness issues.
4. Write findings with expected behavior, actual behavior, and remediation direction.

## Acceptance Criteria

- Tests target real strategy correctness or safety gaps.
- Findings include enough context to reproduce failures.
- No production code is edited.

## Validation

- Run the new strategy audit test file.
- Run selected existing strategy tests if time permits.

## Dependencies

Can run in parallel with W01, W02, and W04.
