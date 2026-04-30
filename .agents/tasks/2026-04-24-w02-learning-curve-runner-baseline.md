# Task ID: 2026-04-24-w02-learning-curve-runner-baseline

## Relation To Overall Task

This creates the first scientific experiment surface: a baseline learning-curve runner/notebook where random learns normally and strategies can be compared under equal label budgets.

## Goal

Add a small, repeatable learning-curve experiment for `synthetic + bert_tiny` that reports per-budget metrics and label efficiency instead of only final endpoint metrics.

## Responsibility Boundaries

Own new benchmark/research artifacts only. Do not change SDK core.

## In Scope

- new script under `benchmarks/`, preferably `benchmarks/run_learning_curve_experiments.py`;
- new notebook if practical, preferably `learning_curve_lab.ipynb` or `lab/learning_curve_lab.ipynb`;
- new checked result artifacts under `benchmarks/results/learning_curves/`;
- docs comments inside the script/notebook where useful.

## Out Of Scope

- SDK engine fixes;
- strategy implementation changes;
- README edits;
- replacing existing benchmark runner.

## Files That May Be Changed

- `benchmarks/run_learning_curve_experiments.py`
- `learning_curve_lab.ipynb` or `lab/learning_curve_lab.ipynb`
- `benchmarks/results/learning_curves/**`
- `.gitignore` only if generated runtime/cache artifacts need ignore coverage

## Files That Must Not Be Touched

- `src/active_learning_sdk/**`
- `tests/**`
- existing benchmark CSVs unless the script intentionally reads them

## Architectural Constraints

- Compare strategies at equal budgets.
- Use baseline recommended by research: `synthetic` dataset, `bert_tiny`, `local` mode first.
- Budgets: `16`, `32`, `48`, `64`, `80`.
- Strategies: `random`, `entropy`, `margin`, `least_confidence`.
- Seeds: use 3 seeds for the first quick artifact.
- Report per-budget rows and aggregate summary.
- Include AULC or normalized mean-over-budget metrics.

## Execution Plan

1. Reuse notebook code where practical rather than duplicating model/dataset logic blindly.
2. Run local learning loops and record metrics after each round/budget.
3. Emit CSV rows with schema including dataset, model, strategy, seed, budget, round, accuracy, macro-F1, duration, selected ids.
4. Emit summary CSV/MD with random baseline, heuristic lifts, and AULC-style means.
5. Create a small notebook wrapper if time permits, using the same output artifacts.
6. Run the quick baseline experiment and report artifacts.

## Acceptance Criteria

- A new learning-curve runner exists.
- Result artifacts exist for `synthetic + bert_tiny`.
- Random baseline reaches at least roughly `accuracy >= 0.45` and `macro_f1 >= 0.40` by 48 labels on average, or the result explicitly reports that the baseline failed.
- Strategy comparisons are budget-matched.
- No SDK core files are changed by this task.

## Expected Validations

- `python -m py_compile benchmarks/run_learning_curve_experiments.py`
- run the quick experiment with `.venv\Scripts\python.exe`
- verify output CSV row counts and no failed runs
