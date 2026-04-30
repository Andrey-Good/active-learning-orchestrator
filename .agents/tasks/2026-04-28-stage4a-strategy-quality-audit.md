# Stage 4A: Strategy Quality Audit

## Context

Stage 1-3 in the current hardening pass closed public contracts, benchmark evidence, and core refactor risks. The next gate is strategy quality: existing strategies must behave like credible active-learning methods, not just pass shape tests.

## Goal

Perform a read-only senior audit of all current strategy implementations and tests. Identify concrete P1/P2 issues where a strategy formula, fallback, tie-break, cold-start guardrail, validation path, scheduler integration, or benchmark-facing behavior is mathematically wrong, misleading, or likely to underperform compared with standard active-learning practice.

## Responsibility Boundaries

Read scope:

- `src/active_learning_sdk/strategies/`
- `src/active_learning_sdk/engine.py` only for scheduler integration and snapshots
- `benchmarks/` only for strategy exposure and metric evidence
- relevant `tests/test_*strategy*.py`, benchmark, and acceptance tests

Write scope:

- `.agents/tmp/2026-04-28-stage4a-strategy-quality-audit.md` only

## In Scope

- Uncertainty formulas: entropy, margin, least-confidence, class/group balanced variants.
- Embedding/diversity methods: CoreSet/k-center, k-means++ style, max-min, density-weighted, dedupe.
- BADGE behavior and cold-start fallback.
- Stochastic and committee formulas: BALD, variation ratio, MC entropy, prediction variance, committee vote entropy, KL, pairwise disagreement, margin.
- Hybrid/adaptive composition and snapshot diagnostics.
- Test/benchmark gaps that can hide quality regressions.

## Out Of Scope

- Do not edit production code or tests.
- Do not propose large new product features unless they are needed to make current strategy behavior correct.
- Do not relitigate already-closed runtime/state/backends defects unless they directly affect strategy quality.

## Architectural Constraints

- Strategies must remain deterministic under identical inputs.
- No strategy may select duplicates or out-of-pool IDs.
- Validation should fail closed on malformed model outputs.
- Strategy quality claims must be backed by tests or benchmark artifacts, not prose alone.

## Execution Plan

1. Read strategy implementations and existing tests.
2. Compare formulas against standard definitions.
3. Look for benchmark fixtures where random/heuristics would be indistinguishable because the fixture is too weak.
4. Produce prioritized findings with exact files/functions and proposed acceptance tests.

## Acceptance Criteria

- Audit file exists.
- Findings are concrete and reproducible.
- If no P1/P2 issues are found, explicitly say what residual risks remain.

## Dependencies

Starts after Stage 3 gates are green.
