# Task ID: 2026-04-24-w04-fix-learning-curve-reproducibility

## Relation To Overall Task

Fix round after independent review of the learning-curve runner.

## Goal

Make learning-curve experiments seed-order reproducible and make the notebook regeneration cell actually runnable.

## Responsibility Boundaries

Own only learning-curve runner/notebook/artifacts.

## In Scope

- `benchmarks/run_learning_curve_experiments.py`
- `lab/learning_curve_lab.ipynb`
- `benchmarks/results/learning_curves/**`

## Out Of Scope

- SDK core;
- existing extended benchmark runner;
- README.

## Review Findings To Address

1. `load_seed_payload()` mutates notebook namespace `GLOBAL_SEED`, and experiments later use whatever global seed was last loaded. Results can depend on `--seeds` order.
2. Notebook markdown says to run the next cell to regenerate artifacts, but the command is commented out.

## Files That May Be Changed

- `benchmarks/run_learning_curve_experiments.py`
- `lab/learning_curve_lab.ipynb`
- `benchmarks/results/learning_curves/**`

## Files That Must Not Be Touched

- `src/active_learning_sdk/**`
- `tests/**`

## Execution Plan

1. Make every run compute seeds from the explicit logical seed, independent of namespace mutation order.
2. Add a validation or small self-check showing seed order does not change per-key output identity.
3. Fix notebook regeneration cell so it is executable as written.
4. Re-run learning-curve artifact generation.
5. Validate py_compile, artifact row counts, no duplicate keys, and notebook validity.

## Acceptance Criteria

- Reversing `--seeds` order produces the same rows after sorting by strategy/seed/budget.
- Notebook can regenerate artifacts from an executable cell.
- Artifacts are regenerated and still honestly report whether random baseline passes.

## Expected Validation

- `python -m py_compile benchmarks/run_learning_curve_experiments.py`
- `.venv\Scripts\python.exe benchmarks\run_learning_curve_experiments.py`
- a reversed-seed reproducibility check
- notebook validation
