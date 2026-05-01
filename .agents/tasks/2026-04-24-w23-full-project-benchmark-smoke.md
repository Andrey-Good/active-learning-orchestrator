# W23 - Full ActiveLearningProject Benchmark Smoke

## Relation To Overall Task
W22 added and R32 approved `ActiveLearningProject.import_labels(...)`, closing the initial-seed API gap. The benchmark harness should now include a smoke path that drives the public project loop rather than only `StrategyScheduler`/`SelectionContext` directly.

## Assumptions And Resolved Ambiguities
- The accepted scheduler-level benchmark remains useful and should not be replaced.
- Full-project smoke is a correctness/integration benchmark, not yet the full strategy matrix.
- It should use the SDK public facade: `ActiveLearningProject.configure(...)`, `import_labels(...)`, and `run(...)`/`run_step(...)`.
- It may use a benchmark-only oracle backend that auto-labels pushed tasks from private labels so the run is deterministic and requires no external service.

## Goal And Expected Result
Add a small full-project smoke benchmark to `benchmarks/sdk_first_benchmark.py` or a companion script under `benchmarks/` that:
- configures an `ActiveLearningProject`;
- imports an initial seed via the new public API;
- runs at least one uncertainty strategy round end-to-end with backend push/poll/pull/train/update;
- emits a compact artifact proving the project loop works with initial labels and no private state mutation.

## Responsibility Boundaries
Owned by this worker:
- `benchmarks/sdk_first_benchmark.py` or new `benchmarks/full_project_smoke.py`
- `benchmarks/README.md`
- generated small artifacts under `benchmarks/results/project_smoke/**`

Do not change:
- `src/active_learning_sdk/**`
- `tests/**`
- root README/docs/docker
- existing smoke scheduler artifacts unless the harness CLI naturally refreshes them and this is documented

## In Scope
- A deterministic oracle backend for benchmark-only use.
- A benchmark-only model adapter if needed.
- Use of `import_labels(...)`.
- Artifact with project status, rounds, selected IDs, final metrics/status counts, and validation report.
- CLI command documented in `benchmarks/README.md`.

## Out Of Scope
- Full matrix migration to project loop.
- SDK algorithm changes.
- External datasets/models.

## Files Or Modules May Be Changed
- `benchmarks/**`

## Files Or Areas Must Not Be Touched
- `src/active_learning_sdk/**`
- `tests/**`
- root `README.md`
- `docs/**`
- `docker/**`
- `.git/**`

## Important Architectural Constraints And Forbidden Actions
- Do not mutate private SDK engine state.
- Do not expose labels to acquisition-visible provider fields.
- Do not fake completed rounds.
- Do not use notebooks.
- Keep artifacts strict JSON.

## High-Level Execution Plan
- Reuse synthetic dataset/model/provider utilities from benchmark harness.
- Build an oracle backend that returns annotations for selected samples using private label map only after push.
- Configure project with explicit train/val split and simulator/custom backend.
- Import seed labels through `project.import_labels(...)`.
- Run to a small budget with entropy or margin.
- Persist `benchmarks/results/project_smoke/summary.json` and `.md`.
- Validate no label leakage and strict JSON.

## Step -> Verify Plan
- Add CLI command/preset -> verify `--help` documents it or new script help works.
- Run project smoke -> verify state counts increase from seed to budget, rounds complete, validation ok.
- Inspect artifacts -> strict JSON parse and no NaN.

## Acceptance Criteria
- Project smoke uses public SDK facade including `import_labels`.
- Project smoke completes at least one active round.
- Artifact proves no fake/private state mutation was used.
- Existing scheduler-level benchmark still runs.

## Expected Tests And Validations
- `python -m py_compile benchmarks/sdk_first_benchmark.py` or new script
- Project smoke command using `.\\.venv\\Scripts\\python.exe`
- Existing scheduler smoke command if changed
- Strict JSON parse for new artifact

## Dependencies
Depends on W22/R32 import labels API.

## Parallel Or Sequential Notes
Sequential before claiming the benchmark harness covers full SDK project loop.
