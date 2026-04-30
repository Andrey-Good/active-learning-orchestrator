# W94 Scheduler, Split, and Cache Blockers

## Context
Senior acceptance review added strict xfail tests for explicit split leakage, bandit scheduler behavior, and invalid probability rows poisoning prediction cache.

## Goal
Fix scheduler/split/cache correctness while preserving existing public behavior where valid.

## Ownership
Primary write scope:
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/strategies/uncertainty.py` only if shared probability validation needs reuse
- `tests/test_senior_acceptance_blockers.py` for converting only relevant xfails into normal passing tests

## In Scope
- Reject duplicate IDs within explicit train/val/test splits.
- Reject overlap across explicit train/val/test splits.
- Implement meaningful `bandit` arm choice for advertised algorithms, at minimum `ucb1`, using reward state.
- Prevent invalid semantic `predict_proba` rows, including zero-sum rows, from being cached.

## Out of Scope
- Backend push/pull atomicity.
- Benchmark fixture/output directory behavior.
- Large active-learning quality experiments.

## Constraints
- Cache validation must happen before cache writes.
- Capability and model errors should remain `ConfigurationError`/`ModelAdapterError` as appropriate.
- Bandit behavior must be deterministic for equal states.
- Do not remove public bandit mode unless implementation is impossible.

## Expected Tests
- `tests/test_senior_acceptance_blockers.py::test_explicit_splits_must_reject_overlap_across_splits`
- `tests/test_senior_acceptance_blockers.py::test_bandit_scheduler_uses_reward_state_instead_of_arm_order`
- `tests/test_senior_acceptance_blockers.py::test_invalid_predict_proba_rows_do_not_poison_prediction_cache`
- Existing strategy capability and audit strategy edge tests.

## Dependencies
Independent of W94 backend and benchmark tasks.
