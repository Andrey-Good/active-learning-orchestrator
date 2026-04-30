# Task W102: Hard Audit Strategies And Cache

## Context

The user requested a hostile senior audit of SDK correctness, including edge cases and hidden bugs. This subtask owns acquisition strategies, scheduler behavior, prediction/embedding cache contracts, and model capability assumptions.

## Goal

Find strategy/cache defects that can produce wrong selections, silent data leakage, nondeterminism, cache poisoning, or misleading benchmark results.

## Responsibility Boundaries

May inspect:

- `src/active_learning_sdk/strategies/**`
- `src/active_learning_sdk/cache.py`
- strategy-related parts of `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/adapters/**`
- relevant tests under `tests/`

May write only:

- `.agents/tmp/w102-strategies-cache-findings.md`

Do not edit source code or tests unless explicitly redirected by the orchestrator.

## In Scope

- Probability validation and class-order correctness
- Embedding shape/type validation
- Duplicate IDs, stale cache keys, split leakage
- Scheduler duplicate/exclusion bugs
- Edge cases: NaN/inf, empty pools, k larger than pool, one-class predictions
- Missing tests that would catch real breakage

## Out Of Scope

- Backend/Label Studio runtime correctness unless it directly corrupts strategy inputs
- Real-dataset benchmark design except where it proves strategy correctness
- Cosmetic style issues

## Architectural Constraints

- Reuse existing tests and fixtures when possible.
- Treat any exact parity with manual formulas as valuable but insufficient if validation gaps remain.
- Work with the dirty worktree as-is.

## Forbidden Actions

- Do not run destructive commands.
- Do not install persistent dependencies.
- Do not modify `uv.lock`, `pyproject.toml`, source files, or tests.
- Do not write benchmark result directories.

## Execution Plan

1. Read strategy/cache/adapters code and relevant tests.
2. Try to falsify selection contracts through adversarial inputs.
3. Identify missing validations or tests.
4. Write findings to `.agents/tmp/w102-strategies-cache-findings.md` with severity, evidence, and repro guidance.

## Acceptance Criteria

- Findings are concrete and tied to behavior, not preference.
- Serious issues include a failing input scenario or precise test that would expose them.
- Clearly distinguish confirmed defects from risks needing broader validation.

## Dependencies

Can run in parallel with W101 and W103. No write-scope overlap.
