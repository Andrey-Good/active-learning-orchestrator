# W94 Backend/Pull State Atomicity

## Context
Senior acceptance review added strict xfail tests for backend task ID validation and pull state corruption. These are release blockers because backend bugs can mutate project state.

## Goal
Fix backend push/pull correctness without changing unrelated scheduler, strategy, or benchmark behavior.

## Ownership
Primary write scope:
- `src/active_learning_sdk/engine.py`
- `tests/test_senior_acceptance_blockers.py` for converting only backend/pull xfails into normal passing tests if the fix is implemented.

## In Scope
- Validate `RoundPushResult.task_ids` exactly matches selected sample IDs before persisting `PUSHED`.
- Make pull atomic: validate the entire backend pull payload before mutating `sample_status`, `sample_labels`, or round status.
- Reject missing annotations for selected tasks instead of marking the round `PULLED`.

## Out of Scope
- Bandit scheduling.
- Explicit split overlap.
- Prediction cache semantic validation.
- Benchmark fixture/output directory behavior.

## Constraints
- Do not weaken label schema validation.
- Do not accept backend annotations for unselected samples.
- Do not mark a round as `PULLED` unless every selected sample has a validated outcome.
- Preserve existing successful project flow tests.

## Expected Tests
- `tests/test_senior_acceptance_blockers.py::test_backend_push_task_ids_must_match_selected_samples`
- `tests/test_senior_acceptance_blockers.py::test_pull_with_later_invalid_label_is_atomic`
- `tests/test_senior_acceptance_blockers.py::test_pull_must_not_complete_when_selected_task_annotations_are_missing`
- Relevant existing runtime audit/project tests.

## Dependencies
Independent of W94 scheduler/cache/benchmark tasks.
