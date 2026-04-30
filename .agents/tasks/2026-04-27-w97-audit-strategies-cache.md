# Task W97-B: Strategies, Selection, Cache Senior Audit

## Context
The user asked for a hard senior-level acceptance review of active-learning quality. The SDK must not merely run; strategies should be correct, robust on edge cases, and comparable to hand-written formulas or known libraries.

## Goal
Perform a read-only audit of strategy implementations, selection normalization, cache behavior, and scoring correctness. Return concrete findings that can become tests or fixes.

## Ownership
Read-only scope:
- `src/active_learning_sdk/strategies/**`
- Strategy-related code in `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/cache.py`
- Strategy/cache tests under `tests/`

Do not edit files.

## In Scope
- Correctness of entropy, margin, least-confidence, random, BADGE, embedding/diversity, hybrid, bandit, stochastic strategies.
- Edge cases: duplicate pool IDs, NaN/Inf/negative probabilities, short model outputs, invalid embeddings, empty pools, k > pool size, one-class outputs.
- Whether strategy normalization hides user strategy bugs.
- Cache key scoping, invalid cache values, cache recovery.
- Parity with direct manual formulas where possible.

## Out of Scope
- Dataset benchmark selection unless it exposes strategy logic problems.
- Backend and state-machine details not tied to selection.

## Special Attention
- Heuristics that silently degrade to random.
- Score direction bugs.
- Selection duplicates or out-of-pool IDs.
- Fallback counters that misreport quality.
- Performance overhead versus direct vectorized formulas.

## Expected Output
- Findings ordered by severity.
- For each finding: file/line, reproduction idea, likely failing test name, and whether it is release-blocking.
- Identify benchmark comparisons that would prove or disprove the issue.
