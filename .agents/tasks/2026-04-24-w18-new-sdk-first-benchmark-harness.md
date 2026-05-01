# W18 - New SDK-First Benchmark Harness

## Relation To Overall Task
The user asked to remove all old notebooks, tests, and benchmark code, then rebuild benchmarks from first principles so SDK algorithm changes can be judged by quality metrics. The old evaluation layer has been removed by W17. R26 produced the benchmark design.

## Assumptions And Resolved Ambiguities
- Old notebooks/tests/benchmark artifacts must stay removed.
- New benchmarks must be scriptable Python artifacts, not notebooks.
- The benchmark must evaluate active learning under a fixed label budget.
- The benchmark should use SDK selection implementations rather than notebook-local strategy logic.
- The current public `ActiveLearningProject` API does not expose a clean way to import an initial labeled seed before uncertainty selection. Do not hack private engine state. If full-project execution is not clean yet, document this as an SDK gap and implement the core benchmark around public SDK strategy/scheduler APIs.
- Generated benchmark result files are allowed only if they are small smoke artifacts useful for verifying the harness.

## Goal And Expected Result
Create a new benchmark harness that can compare random, uncertainty heuristics, and mixed strategies on deterministic small synthetic datasets. It must produce numeric artifacts that show learning curves, lift versus random, runtime, selection diagnostics, and budget efficiency.

## Responsibility Boundaries
Owned by this worker:
- New `benchmarks/` directory and its files.
- Small generated benchmark result artifacts under `benchmarks/results/` if produced by a smoke run.
- Optional benchmark-only documentation under `benchmarks/README.md`.
- `pyproject.toml` only if needed to remove notebook-specific product dependencies or add benchmark script metadata without changing SDK behavior.

Do not change:
- `src/active_learning_sdk/**`
- `README.md`
- `docs/**`
- `docker/**`
- unrelated generated caches or lockfiles unless a dependency manifest change requires it.

## In Scope
- A runnable CLI benchmark script.
- Deterministic synthetic text classification datasets that test different active-learning failure/success modes.
- Fast scikit-learn model adapter used only by the benchmark.
- Metrics: accuracy, macro-F1, weighted-F1, rare-class recall when applicable, AULC, lift vs random, selected label distribution, duplicate/group concentration diagnostics, runtime.
- Budgeted active-learning curves at budgets such as 16/32/48/64/96.
- Strategy configs: random, entropy, margin, least_confidence, mix entropy/random, mix uncertainty/random.
- A smoke command that finishes quickly.
- Durable documentation of benchmark design and current SDK API limitation around initial labeled seed.

## Out Of Scope
- SDK algorithm changes.
- Full product README rewrite.
- Reintroducing notebooks.
- Reintroducing `tests/` in this task.
- Downloading large external datasets or neural networks.

## Files Or Modules May Be Changed
- `benchmarks/**`
- `pyproject.toml` only if justified by the benchmark layer cleanup.

## Files Or Areas Must Not Be Touched
- `src/active_learning_sdk/**`
- `README.md`
- `docs/**`
- `docker/**`
- `.git/**`
- Existing `.agents/tasks/**` except this task document.

## Architectural Constraints And Forbidden Actions
- Do not implement strategy formulas independently when SDK built-ins can be invoked.
- Do not mutate private SDK engine state to seed labels.
- Do not rely on notebooks.
- Do not make benchmark success depend on external services.
- Keep runtime small enough for local iteration; default smoke should run in under a few minutes.
- If `ActiveLearningProject` cannot be used cleanly for the main matrix, state why in `benchmarks/README.md`.

## High-Level Execution Plan
- Inspect SDK scheduler/context interfaces enough to call them correctly.
- Create benchmark datasets with explicit metadata for class/group/rare-class diagnostics.
- Create a benchmark-only sklearn text model adapter.
- Implement active-learning loop with a transparent initial seed policy and SDK scheduler calls for acquisition.
- Persist CSV/JSON/Markdown artifacts.
- Add a smoke CLI preset and run it.
- Remove notebook dependencies from `pyproject.toml` if they are only leftovers from old notebooks and verify environment still resolves.

## Step -> Verify Plan
- Implement CLI and dataset/model helpers -> run `python benchmarks/sdk_first_benchmark.py --help`.
- Implement strategy matrix and metrics -> run a tiny smoke suite with 1-2 datasets, 2-3 strategies, and small budgets.
- Persist artifacts -> inspect result directory contains manifest, metrics, selections, summary JSON/MD.
- Document design -> verify `benchmarks/README.md` names datasets, metrics, strategy configs, and known SDK gap.
- If editing `pyproject.toml` -> run `uv lock --check` if possible or report if lock update is needed.

## Acceptance Criteria
- No notebooks are recreated.
- The harness is deterministic for fixed seeds.
- It compares at least random, entropy, margin, least_confidence, and one mix strategy.
- It emits budgeted metrics and lift-vs-random values.
- It can be run from the repository root with a documented command.
- It clearly distinguishes benchmark-only initial seed logic from SDK product API.
- Smoke run completes successfully and writes small artifacts.

## Expected Tests And Validations
- `python benchmarks/sdk_first_benchmark.py --help`
- A smoke benchmark command with small matrix.
- Basic artifact existence/readability checks.
- If dependency metadata changes, appropriate lock/check command or explicit note.

## Dependencies
- Depends on W17 cleanup.
- Incorporates R26 benchmark design.
- Later SDK improvement tasks will consume this harness and its metrics.

## Parallel Or Sequential Notes
Sequential before SDK algorithm rewrites. Algorithm workers should not start until this benchmark harness exists and has at least one smoke run.
