# Task W97-C: Benchmarks and Quality Evidence Audit

## Context
The user wants benchmarks showing whether this SDK is better than random selection, where it is worse than manual formulas or analogs, and whether reported quality evidence is honest.

## Goal
Audit benchmark harnesses and existing benchmark results. Identify missing benchmarks, weak fixtures, misleading metrics, or places where the SDK is slower/worse than it should be.

## Ownership
Read-only scope:
- `benchmarks/**`
- Benchmark-related tests under `tests/`
- Benchmark result summaries under `benchmarks/results/**`
- Relevant README/docs benchmark claims

Do not edit files.

## In Scope
- Manual formula parity and runtime overhead.
- Budgeted active learning quality metrics.
- Dataset/model/strategy coverage.
- Stop criteria speed-quality tradeoff.
- Reproducibility: seeds, manifests, validation files.
- Whether benchmark artifacts support README-grade claims.

## Out of Scope
- Implementing new benchmark code.
- SDK algorithm fixes.

## Special Attention
- Benchmarks that pass but are too weak to detect regressions.
- Tiny fixtures where every heuristic chooses the same order.
- Missing analog comparison metadata.
- Metrics that hide variance or class imbalance.
- Results that are stale or inconsistent with current code.

## Expected Output
- Findings ordered by severity.
- Concrete benchmark additions or modifications to make evidence senior-grade.
- Any current benchmark numbers that look suspicious or unacceptable.
