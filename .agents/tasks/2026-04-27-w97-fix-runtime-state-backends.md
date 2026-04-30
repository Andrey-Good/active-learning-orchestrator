# Task W97-F: Fix Runtime, State, Backend Integrity Blockers

## Context
Senior audit found runtime/state/backend release blockers:
- swapped backend task-id values can silently assign labels to wrong samples;
- all-NEEDS_REVIEW pulls advance to train/eval and can brick the round;
- multiple active rounds can orphan earlier backend tasks;
- `sample_status` coverage can diverge from attached dataset IDs;
- reconfigure can change label schema while preserving incompatible labels;
- in-memory built-in backends cannot resume pushed rounds;
- stale lock files block crash recovery.

## Goal
Implement focused fixes for these runtime integrity problems without weakening existing tests.

## Ownership
Allowed write scope:
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/state/store.py`
- `src/active_learning_sdk/state/lock.py`
- `src/active_learning_sdk/backends/base.py`
- `src/active_learning_sdk/backends/simulator.py`
- focused tests in `tests/test_w97_senior_audit_acceptance.py`

Do not edit:
- strategy math files;
- benchmark harnesses;
- README/docs except later audit report references.

## Requirements
- Detect task-id/sample-id mismatch for built-in backends before returning progress/annotations.
- Ensure all-NEEDS_REVIEW pull completes safely without entering train/eval when no labels were resolved.
- Validate that at most one active round exists.
- Validate attached dataset IDs exactly match `sample_status` keys once configured.
- Reject label schema changes when existing labels or rounds would become incompatible.
- For in-memory LLM/simulator backends, either make resume work from deterministic task ids and current dataset/backend oracle, or fail early with a clear non-durable backend error before corrupting labels.
- Treat stale lock files for dead local PIDs as recoverable.

## Acceptance
- Focused W97 tests pass.
- Existing full suite passes.
- No silent fallback that hides data-integrity degradation.
