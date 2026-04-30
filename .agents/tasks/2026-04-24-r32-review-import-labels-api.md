# R32 - Review SDK Import Labels API

## Relation To Overall Task
W22 added `ActiveLearningProject.import_labels(...)` / engine import support to close the initial-seed API gap. This review gates using it in benchmarks and further SDK improvements.

## Assumptions And Resolved Ambiguities
- Imported labels are external/manual/oracle labels and should not create fake backend rounds.
- Default overwrite protection is required.
- Unknown sample IDs and invalid labels must fail without partial state mutation.
- Tests were added by W22 and reportedly pass.

## Goal And Expected Result
Perform read-only review of W22 changes. Check API correctness, edge cases, state persistence, validation behavior, tests, and integration with existing run/resume semantics.

## Responsibility Boundaries
Read-only. Do not edit files.

## In Scope
- `src/active_learning_sdk/engine.py` import label implementation and helpers.
- `src/active_learning_sdk/project.py` facade.
- `tests/test_import_labels.py`.
- Whether validation is atomic before state mutation.
- Whether multi-label validation matches current `LabelSchema.multi_label` semantics.
- Whether idempotent re-import and overwrite behavior are correct.
- Whether source/summary return is useful and stable.

## Out Of Scope
- Benchmark harness migration to full project loop.
- SDK strategy changes.
- README rewrite.
- Dependency cleanup.

## Files Or Modules May Be Changed
None.

## Files Or Areas Must Not Be Touched
Entire repository.

## Important Architectural Constraints And Forbidden Actions
- Do not require fake rounds.
- Do not require state-only import without runtime unless you can show current API contract already supports similar state-only mutations.
- Separate optional follow-ups from blocking findings.

## High-Level Execution Plan
- Inspect code diff/implementation.
- Inspect tests.
- Optionally rerun targeted tests if cheap.
- Report concrete findings or explicit approval.

## Acceptance Criteria
- No in-scope blocking defects/risks/questions/required improvements remain.
- If issues remain, they are concrete and tied to product behavior.

## Expected Tests And Validations
Optional:
- `uv run --group dev pytest tests/test_import_labels.py -q`

## Dependencies
Depends on W22.

## Parallel Or Sequential Notes
Sequential before benchmark migration uses the API.
