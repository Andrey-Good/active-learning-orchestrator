# Task w86: Quality Improvement Implementation

## Context
After diagnosis, the main orchestrator will implement SDK changes to materially improve active-learning benchmark metrics. This worker task is reserved for implementation once the exact patch scope is known.

## Goal
Implement narrowly scoped product-grade SDK improvements that improve active-learning quality on real/synthetic benchmarks while preserving deterministic behavior and tests.

## Responsibility Boundaries
Wait for explicit instructions from the orchestrator before editing. Expected ownership may include strategy files and tests only.

## In Scope
- Strategy-level improvements such as exploration/diversity guardrails, normalized embeddings, better BADGE initialization, or scheduler fixes.
- Corresponding unit tests.

## Out of Scope
- Do not edit README unless explicitly instructed.
- Do not alter dataset labels or leak oracle labels.
- Do not weaken tests.

## Files May Be Changed After Assignment
- To be specified by orchestrator before worker starts.

## Files Must Not Be Touched
- Any file outside the assigned scope.

## Constraints
- Deterministic decisions.
- No benchmark-specific hardcoding.
- Existing public API should remain compatible unless the orchestrator approves an additive extension.

## Plan
1. Wait for orchestrator instructions.
2. Implement the assigned change.
3. Add focused tests.
4. Run targeted tests and report changed files.

## Acceptance Criteria
- Tests pass for touched functionality.
- Benchmarks improve on at least one real scenario without catastrophic regressions.

## Dependencies
Depends on r105/r106 diagnosis or direct orchestrator assignment.
