# W27 - Remove Project Smoke External Warm-Start

## Relation To Overall Task
The SDK now supports seed train/eval before first SELECT, and R36 accepted the implementation. The full project smoke benchmark should stop externally warm-starting the model and instead prove the public SDK state machine handles imported seed labels itself.

## Assumptions And Resolved Ambiguities
- W24-W26/R36 accepted SDK seed train before first select.
- Project smoke should now expect first `run_step` after `import_labels(...)` to be `TRAIN_EVAL` with `round_id=None`, followed by a real active SELECT/PUSH/WAIT/PULL/TRAIN_EVAL/UPDATE round.
- No benchmark private state mutation or direct model warm-start should remain.

## Goal And Expected Result
Update project smoke so:
- it does not call `model.fit(...)` directly before selection;
- it records `model_warm_started_from_imported_seed: false`;
- it records/validates a public seed train step before SELECT;
- generated project smoke artifacts are refreshed.

## Responsibility Boundaries
Owned by this worker:
- `benchmarks/sdk_first_benchmark.py`
- `benchmarks/README.md`
- `benchmarks/results/project_smoke/**`

Do not change:
- `src/active_learning_sdk/**`
- `tests/**`
- scheduler smoke artifacts unless the command naturally updates only project smoke
- root README/docs/docker

## In Scope
- Remove benchmark model warm-start workaround.
- Update project-smoke validation/artifact fields.
- Rerun project smoke.
- Strict JSON validation.

## Out Of Scope
- SDK changes.
- Full benchmark matrix migration.
- Strategy changes.

## Files Or Modules May Be Changed
- `benchmarks/**`

## Files Or Areas Must Not Be Touched
- `src/active_learning_sdk/**`
- `tests/**`
- `README.md`
- `docs/**`
- `docker/**`

## Important Architectural Constraints And Forbidden Actions
- Do not call direct `model.fit(...)` before SDK `run_step`.
- Do not mutate private engine state.
- Do not fake seed TRAIN_EVAL in artifacts; it must come from actual `run_step`.
- Keep JSON strict.

## High-Level Execution Plan
- Remove warm-start code from `run_project_smoke`.
- Update step loop/validation to expect seed TRAIN_EVAL then active round completion.
- Update README language.
- Regenerate project smoke artifacts.

## Step -> Verify Plan
- Modify project smoke -> run `.\\.venv\\Scripts\\python.exe benchmarks/sdk_first_benchmark.py --preset project_smoke`.
- Verify summary shows `model_warm_started_from_imported_seed: false`, `private_state_mutation_used: false`, and includes seed train step.
- Strict JSON parse.

## Acceptance Criteria
- Project smoke proves public SDK seed train behavior.
- No SDK files changed.
- Artifacts refreshed and valid.

## Expected Tests And Validations
- `.\\.venv\\Scripts\\python.exe -m py_compile benchmarks/sdk_first_benchmark.py`
- `.\\.venv\\Scripts\\python.exe benchmarks/sdk_first_benchmark.py --preset project_smoke`
- Strict JSON parse/no NaN for project smoke artifacts

## Dependencies
Depends on R36.

## Parallel Or Sequential Notes
Sequential before further algorithm benchmark iterations.
