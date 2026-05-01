# W22 - SDK Import Labels / Initial Seed API

## Relation To Overall Task
The accepted benchmark harness documents a core SDK gap: `ActiveLearningProject` lacks a clean public way to import an initial labeled seed before uncertainty selection. This prevents a fully SDK-driven active-learning benchmark and is a real product limitation for cold-start active learning.

## Assumptions And Resolved Ambiguities
- The benchmark harness is accepted by R31.
- This task should add a small, public, reliable API rather than mutate private state.
- Imported labels represent external/oracle/manual labels, not backend round annotations. They should update `sample_status` and `sample_labels`, but they should not create fake rounds/tasks.
- Default behavior must protect users from accidental label overwrites.
- The API must validate sample IDs and label values against the configured dataset and `LabelSchema`.

## Goal And Expected Result
Add a public `ActiveLearningProject.import_labels(...)` method (or equivalently named method if a better name is already present) that imports known labels into project state so the next active-learning selection round can train on seed labels and use uncertainty strategies correctly.

## Responsibility Boundaries
Owned by this worker:
- `src/active_learning_sdk/project.py`
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/__init__.py` only if a new public return type/config is introduced
- `tests/**` for focused tests of the new API
- Adjacent docs under `benchmarks/README.md` only if needed to mention the new API as future integration path

Do not change:
- `benchmarks/sdk_first_benchmark.py` in this task
- generated benchmark artifacts
- Docker/Label Studio integration
- unrelated SDK strategy algorithms

## In Scope
- Public method on project facade and engine.
- Input as mapping `{sample_id: label}`.
- `overwrite=False` default; conflicting labels raise a clear SDK error.
- Idempotent re-import of the same labels should not fail.
- Unknown sample IDs should fail.
- Invalid labels should fail.
- Multi-label validation if current `LabelSchema.multi_label=True`.
- Return a small summary dict: imported/unchanged/overwritten/total counts and maybe source.
- Persist state atomically.
- Tests for success, idempotency, unknown sample, invalid label, overwrite conflict, overwrite allowed, and effect on status counts.

## Out Of Scope
- Full benchmark harness migration to `ActiveLearningProject` loop.
- Backend annotation history/audit trails for imported labels.
- CLI interface.
- SDK strategy quality changes.

## Files Or Modules May Be Changed
- `src/active_learning_sdk/project.py`
- `src/active_learning_sdk/engine.py`
- `tests/**`
- Optional: `src/active_learning_sdk/__init__.py` only if needed

## Files Or Areas Must Not Be Touched
- `benchmarks/results/**`
- `benchmarks/sdk_first_benchmark.py`
- `docker/**`
- root `README.md`
- `docs/**`
- `.git/**`

## Important Architectural Constraints And Forbidden Actions
- Do not create fake completed rounds for imported labels.
- Do not bypass dataset fingerprint/configuration checks.
- Do not silently overwrite conflicting labels.
- Do not accept labels outside `LabelSchema.labels`.
- Do not mutate returned `ProjectState` externally; add a real engine method.
- Preserve existing run/resume semantics.

## High-Level Execution Plan
- Add engine method `import_labels(...)`.
- Add facade method on `ActiveLearningProject`.
- Implement helper validation for single-label and multi-label.
- Add focused pytest tests with minimal in-memory dataset/model/backend stubs.
- Run targeted tests and broader available checks.

## Step -> Verify Plan
- Implement API -> test import updates status and persisted state.
- Add validation -> test unknown sample and invalid label fail.
- Add overwrite behavior -> test conflict fails by default, succeeds with `overwrite=True`.
- Check integration -> configure project with seed labels, then status shows labeled count before run.

## Acceptance Criteria
- Users can import initial labels via public API after `configure(...)`.
- Imported labels survive reopening the same workdir.
- Validation errors are clear and deterministic.
- Tests pass.
- No benchmark artifacts are changed in this task.

## Expected Tests And Validations
- `uv run --group dev pytest tests/test_import_labels.py -q`
- If feasible: `uv run --group dev pytest -q`

## Dependencies
Depends on accepted benchmark harness R31.

## Parallel Or Sequential Notes
Sequential first SDK improvement. Later tasks can update benchmark harness to use this API for a full `ActiveLearningProject` smoke.
