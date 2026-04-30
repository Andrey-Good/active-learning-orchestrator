# 2026-04-28 W113 Runtime/API Acceptance Audit

## Context
The user requested a senior-level acceptance audit of the current SDK state, with real defects, stress tests, reproducible evidence, and no cosmetic complaints. This subtask covers public API, engine/runtime state, dataset/split handling, annotation lifecycle, adapters, and packaging boundaries.

## Goal
Find all justified correctness, reliability, maintainability, API, and packaging objections in the runtime/API surface. Prefer executable evidence: failing or xfail tests, minimal repros, command outputs, and concrete file/line references.

## Ownership
Read scope: `src/active_learning_sdk/**/*.py`, `tests/**/*.py`, `pyproject.toml`, `README.md`, existing audit docs.
Write scope: only `.agents/tmp/2026-04-28-w113-runtime-api-findings.md`.

## In Scope
- Public API exports and optional dependency behavior.
- Engine lifecycle, state restore/resume, locking, splits, annotation timeout/pull/push flows.
- Adapter contracts and model capability validation.
- Packaging metadata and import hygiene.

## Out Of Scope
- Do not edit production code.
- Do not edit tests, benchmarks, docs, or pyproject.
- Do not run destructive git commands.
- Do not touch files outside the write scope.

## Constraints
- Treat existing uncommitted changes as user work.
- Findings must be evidence-backed, not taste-based.
- Mark severity and explain why the issue matters after one or two years of maintenance.

## Execution Plan
1. Inspect relevant source and current tests.
2. Run targeted commands if useful, keeping outputs summarized in the findings file.
3. Identify confirmed defects and residual high-risk gaps.
4. Write a concise findings file with reproduction notes.

## Acceptance Criteria
- Findings file exists at `.agents/tmp/2026-04-28-w113-runtime-api-findings.md`.
- Each finding includes affected file(s), severity, evidence, and suggested acceptance test/fix direction.
- No production/test/benchmark files are modified.

## Dependencies
Can run in parallel with W114 and W115. Final synthesis depends on this report.
