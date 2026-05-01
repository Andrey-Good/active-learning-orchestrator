# Task fix03 - custom selector duplicate/out-of-pool contract

## Context
The black-box stress review found that a custom selector can return duplicate sample ids. The SDK currently normalizes the selection, silently shrinks the batch, and proceeds instead of failing as an SDK contract violation. Previous audits also found out-of-pool ids should surface as strategy-level failures for custom strategy output.

## Goal
Make custom selector output strict: duplicate ids, selected ids outside the candidate pool, or malformed ids must raise `StrategyError` before any round is pushed.

## Ownership
Allowed changes: `src/active_learning_sdk/engine.py`, focused tests under `tests/`.
Must not change built-in strategy behavior except via shared validation if tests prove it is compatible.

## Acceptance Criteria
- Duplicate ids returned by a custom selector raise `StrategyError`.
- Out-of-pool ids returned by a custom selector raise `StrategyError`.
- Valid custom selector output still works.
- Existing built-in strategy tests remain green.

## Validation
Run focused custom strategy edge tests and any existing scheduler tests that cover custom selectors.
