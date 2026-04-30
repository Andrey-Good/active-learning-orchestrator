# 2026-04-28 W115 Benchmark/Reference Acceptance Audit

## Context
The user explicitly asked for benchmarks showing where this SDK is worse than analogs or manual work, and allowed installing modAL / skactiveml if useful without making a mess. The repository already contains benchmark scripts and historical results, so this subtask must validate the current benchmark story rather than just trust old artifacts.

## Goal
Audit benchmark quality, external-reference comparisons, result reproducibility, evidence hygiene, and claims in docs/README.

## Ownership
Read scope: `benchmarks/**/*.py`, `benchmarks/results/**`, `tests/test_*benchmark*.py`, `docs/**/*.md`, `README.md`, `pyproject.toml`.
Write scope: only `.agents/tmp/2026-04-28-w115-benchmark-reference-findings.md`.

## In Scope
- Benchmark scripts and result schema.
- Comparisons to manual/reference implementations and optional external libraries.
- Reproducibility, seed control, leakage risks, metric validity.
- Whether docs overclaim beyond measured evidence.

## Out Of Scope
- Do not edit production code, benchmark scripts, tests, docs, or dependency files.
- Do not perform heavy downloads or long external benchmark runs.
- Do not leave package/environment clutter outside the existing project environment.

## Constraints
- If external packages are unavailable, document that and assess existing adapter hooks/results.
- Findings must distinguish benchmark-code defects from “missing stronger evidence.”

## Execution Plan
1. Inspect benchmark scripts/results and current tests.
2. Run light smoke/validation commands if feasible.
3. Record defects, gaps, and proposed acceptance benchmark cases.

## Acceptance Criteria
- Findings file exists at `.agents/tmp/2026-04-28-w115-benchmark-reference-findings.md`.
- Report includes concrete benchmark weaknesses and how to reproduce/strengthen them.

## Dependencies
Can run in parallel with W113 and W114. Final synthesis depends on this report.
