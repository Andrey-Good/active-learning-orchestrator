# R39 - Review Group-Diverse Entropy Strategy

## Relation To Overall Task
W29 implemented `group_diverse_entropy` as the first SDK acquisition improvement motivated by the accepted baseline matrix. This review gates benchmark integration.

## Assumptions And Resolved Ambiguities
- Strategy should use group IDs only, not labels.
- Determinism and unique clipped output are required.
- Fixed one-per-group first pass is acceptable for first implementation.

## Goal And Expected Result
Review W29 changes for correctness, architectural fit, edge cases, and test adequacy. Explicitly state whether any in-scope blocking defects/risks/questions/required improvements remain.

## Responsibility Boundaries
Read-only. Do not edit files.

## In Scope
- `src/active_learning_sdk/strategies/uncertainty.py`
- `src/active_learning_sdk/engine.py` registration/lookup
- `src/active_learning_sdk/strategies/__init__.py`
- `tests/test_group_diverse_strategy.py`
- Existing tests impact

## Out Of Scope
- Benchmark harness registration/runs.
- Configurable group caps.
- Embedding diversity.

## Files Or Modules May Be Changed
None.

## Files Or Areas Must Not Be Touched
Entire repository.

## Important Architectural Constraints And Forbidden Actions
- Do not require configurable behavior unless fixed behavior is defective.
- Keep optional follow-ups separate from blockers.

## High-Level Execution Plan
- Inspect implementation and tests.
- Optionally rerun tests.
- Report findings and status.

## Acceptance Criteria
- Strategy is correct enough to benchmark.
- No blocking issues remain.

## Expected Tests And Validations
Optional:
- `uv run --group dev pytest tests/test_group_diverse_strategy.py -q`

## Dependencies
Depends on W29.

## Parallel Or Sequential Notes
Sequential before adding strategy to benchmark matrix.
