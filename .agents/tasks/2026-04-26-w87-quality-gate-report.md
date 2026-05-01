# Task w87: Quality Gate Report Implementation

## Context
The orchestrator will add or use an aggregator for multi-seed/multi-budget benchmark outputs. This task document records scope for implementation/review.

## Goal
Create a repeatable report that summarizes active-learning benchmark quality, not just raw rows.

## Responsibility Boundaries
Expected write scope, if delegated: `benchmarks/quality_gate_report.py`, tests under `tests/`.

## In Scope
- Parse `metrics.csv` and optional `full_train_reference.csv`.
- Compute mean/std, strategy-vs-random lift, win rate, normalized AULC, coverage/zero-recall, runtime.
- Emit strict JSON and Markdown.
- Define pass/fail quality gates.

## Out of Scope
- Do not change SDK strategies.
- Do not change dataset generation.

## Constraints
- Deterministic.
- Strict JSON serializable.
- Works on incomplete strategy/dataset grids.

## Acceptance Criteria
- Tests cover aggregation math and missing-random handling.
- Output is readable enough to paste into README/report.

## Dependencies
Depends on chosen protocol.
