# Task W98-E: Review Strategy Probability Cube And Cache Identity Fixes

## Context
W98-B fixed one-column stochastic/committee probability cubes and cache scope type identity.

## Goal
Read-only review: decide whether real issues remain in this scope.

## Scope
Read only:
- `src/active_learning_sdk/strategies/stochastic.py`
- `src/active_learning_sdk/cache.py`
- `tests/test_acceptance_strategy_correctness_2026_04_27.py`
- relevant existing strategy/cache tests

Do not edit files.

## Known Validation
- Acceptance with `--runxfail` -> `16 passed`
- Focused adjacent set -> `95 passed`
- Full suite -> `370 passed`

## Review Questions
- Do stochastic/committee rows now require at least two probability columns?
- Does the validation remain consistent with the uncertainty probability contract?
- Do cache keys distinguish `1`, `"1"`, `"int:1"`, `None`, and `"none:null"`?

## Output
Return findings ordered by severity. If no issues remain, say so clearly.
