# 2026-04-28 W114 Strategy/Cache Stress Audit

## Context
The user requested a hard audit that attempts to break the SDK, especially hidden correctness and edge-case failures. This subtask focuses on strategy correctness, probability/embedding validation, caches, deterministic tie-breaking, duplicate ids, stochastic/committee behavior, and scalability risks.

## Goal
Find justified objections in strategy and cache behavior, with concrete repro ideas and stress-test candidates.

## Ownership
Read scope: `src/active_learning_sdk/strategies/**/*.py`, `src/active_learning_sdk/cache.py`, `src/active_learning_sdk/engine.py`, related tests and benchmark scripts.
Write scope: only `.agents/tmp/2026-04-28-w114-strategy-cache-findings.md`.

## In Scope
- Active learning strategy semantics and correctness.
- Cache keying/invalidation/stale data risks.
- Determinism, duplicate handling, input validation, malformed probability/embedding rows.
- Complexity and scalability issues.

## Out Of Scope
- Do not edit production code.
- Do not edit tests or benchmarks.
- Do not install packages unless necessary for a small local repro.

## Constraints
- Findings must be actionable and evidence-backed.
- Prefer focused repros over broad speculation.
- Include “no finding” notes only when a historically risky area was checked and passed.

## Execution Plan
1. Inspect strategy/cache implementations and existing edge-case tests.
2. Run targeted tests or small snippets for suspected failures.
3. Record confirmed findings and high-value missing tests.

## Acceptance Criteria
- Findings file exists at `.agents/tmp/2026-04-28-w114-strategy-cache-findings.md`.
- Each finding includes severity, evidence, expected vs actual behavior, and suggested test/fix.

## Dependencies
Can run in parallel with W113 and W115. Final synthesis depends on this report.
