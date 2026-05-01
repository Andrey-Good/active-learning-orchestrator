# R33 - Review Full Project Benchmark Smoke

## Relation To Overall Task
W23 added a full `ActiveLearningProject` smoke benchmark after the import-labels API was approved. This review checks whether the project smoke is methodologically sound enough to prove public project-loop integration and identify any remaining SDK gap.

## Assumptions And Resolved Ambiguities
- W23 was allowed to edit `benchmarks/**` only.
- Full project smoke is an integration smoke, not a full strategy-quality matrix.
- The project currently selects before its first train/eval step, so W23 reportedly warm-starts the model from imported seed labels before selection without mutating private SDK state.

## Goal And Expected Result
Perform read-only review of W23 changes and artifacts. Decide whether the full-project smoke can be accepted, and classify the warm-start behavior as acceptable workaround, required benchmark fix, or SDK product gap.

## Responsibility Boundaries
Read-only. Do not edit files.

## In Scope
- `benchmarks/sdk_first_benchmark.py` project-smoke path.
- `benchmarks/README.md` project-smoke documentation.
- `benchmarks/results/project_smoke/**`.
- Whether public API is used (`configure`, `import_labels`, `run_step`/`run`) and private SDK state is not mutated.
- Whether oracle labels are exposed to acquisition-visible provider fields.
- Whether artifacts are strict JSON and meaningful.

## Out Of Scope
- SDK algorithm implementation changes.
- Full benchmark migration.
- Root README rewrite.
- Dependency cleanup.

## Files Or Modules May Be Changed
None.

## Files Or Areas Must Not Be Touched
Entire repository.

## Important Architectural Constraints And Forbidden Actions
- Do not require project smoke to be a quality benchmark; it is an integration proof.
- Do not reject warm-start solely because it exists; classify whether it is clearly documented and not private-state mutation.
- Keep optional follow-ups separate from blockers.

## High-Level Execution Plan
- Inspect project smoke code path and artifacts.
- Verify public API usage and no private SDK state mutation.
- Verify strict JSON and project validation result.
- Report findings and acceptance status.

## Acceptance Criteria
- Smoke uses public facade and completes a real round.
- Artifacts record seed import, selection, status counts, and validation.
- No blocking benchmark methodology defect remains for integration smoke.

## Expected Tests And Validations
Optional:
- `python -m py_compile benchmarks/sdk_first_benchmark.py`
- strict JSON parse of project smoke artifacts
- inspect `state.json`

## Dependencies
Depends on W23.

## Parallel Or Sequential Notes
Sequential gate before using project smoke as evidence.
