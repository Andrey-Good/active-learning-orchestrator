# Task W98-B: Strategy Probability Cubes And Cache Key Identity

## Context
Stress review added strict xfail probes in `tests/test_acceptance_strategy_correctness_2026_04_27.py`.

## Goal
Turn strategy/cache xfails green by fixing production behavior, not weakening tests.

## Ownership
May change:
- `src/active_learning_sdk/strategies/stochastic.py`
- `src/active_learning_sdk/cache.py`
- `tests/test_acceptance_strategy_correctness_2026_04_27.py`
- narrowly related existing strategy/cache tests

Must not change:
- runtime state validation
- public split/prelabel behavior
- benchmark harnesses

## Problems To Fix
1. Stochastic and committee probability cube validation accepts one-column probability rows.
2. Prediction and embedding cache scoped keys alias scalar `1` and string `"1"` because scope parts are stringified before hashing.

## Expected Behavior
- Probability cube/member rows must follow the same probability contract as ordinary uncertainty strategies: at least two probability columns, finite, non-negative, and sum to 1.0.
- Cache key scoping must include type tags for scalar scope parts so `1` and `"1"` are distinct.
- Existing string-only ID behavior should remain stable for normal provider paths as much as possible.

## Acceptance Criteria
- Focused strategy acceptance tests pass without xfail.
- Existing strategy/cache tests pass.
- No broad rewrite of unrelated strategy logic.
