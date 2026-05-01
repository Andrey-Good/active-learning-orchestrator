# W17 - Remove Old Evaluation Layer

## Relation to Overall Task
The user requested removing all old notebooks and old code for checks/tests/benchmarks before rebuilding benchmarks correctly.

## Goal
Delete the old evaluation layer while preserving product SDK code.

## In Scope
Delete:
- all `*.ipynb` files;
- `benchmarks/**`;
- `tests/**`;
- generated experiment CSVs that are old benchmark outputs:
  - `experiment_runs.csv`;
  - `lab/experiment_runs.csv`.

Update config only if it references deleted test paths.

## Out of Scope
- Do not delete `src/**`.
- Do not delete Docker/backend product files.
- Do not create the new benchmark harness in this task.
- Do not modify README.

## Files/Areas May Change
- delete `active_learning_lab.ipynb`;
- delete `lab/active_learning_lab.ipynb`;
- delete `lab/learning_curve_lab.ipynb`;
- delete `benchmarks/**`;
- delete `tests/**`;
- delete old experiment CSV outputs;
- `pyproject.toml` only if needed to remove obsolete pytest config.

## Files/Areas Must Not Touch
- `src/active_learning_sdk/**`;
- `docker/**`;
- `docs/**`;
- `README.md`.

## Safety Constraints
- Resolve absolute paths and verify all deletion targets are inside `C:\Dev\active-learning-orchestrator_with_agent`.
- Do not use git reset/checkout.
- Do not delete `.agents/tasks`.

## Acceptance Criteria
- No notebooks remain.
- Old `benchmarks` and `tests` directories are gone.
- No pyproject reference points to missing test directory.
- Final report lists deleted paths and any config updates.

## Validation
- Directory/file existence checks.
- `git status --short` review.
