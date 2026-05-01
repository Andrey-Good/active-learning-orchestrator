# R10 - Baseline Stability Hypotheses

## Relation to Overall Task
This is a read-only research task for the scientific optimization cycle. The current learning-curve artifact shows that `synthetic + bert_tiny` does not give a stable random baseline at 48 labels, so strategy comparisons are not yet reliable.

## Assumptions and Resolved Ambiguities
- We are not adding new active-learning methods yet.
- Existing strategy and engine behavior must be measured before expanding the SDK.
- Active learning quality must be evaluated under label budgets, not only final full-data quality.
- Random should be a usable baseline, but not saturated, so heuristic lift remains measurable.

## Goal
Produce a concise experimental plan for choosing a stable baseline dataset/model/budget configuration.

## Responsibility Boundaries
- Read-only.
- Do not edit repository files.
- Inspect existing benchmark artifacts, notebook configuration, and benchmark runners.

## In Scope
- Identify likely causes of poor random baseline stability.
- Propose 3-6 concrete hypotheses that can be tested quickly.
- Define acceptance metrics for a "good baseline" and a "useful active-learning benchmark".
- Recommend a small sweep matrix that should run quickly.

## Out of Scope
- No code changes.
- No long training runs.
- No new SDK methods such as BADGE.

## Files/Areas May Read
- `active_learning_lab.ipynb`
- `lab/learning_curve_lab.ipynb`
- `benchmarks/run_learning_curve_experiments.py`
- `benchmarks/results/learning_curves/*`
- Existing docs/plans if needed.

## Files/Areas Must Not Touch
- All files are read-only for this task.

## Architectural Constraints and Forbidden Actions
- Do not propose hiding bad results by changing metrics.
- Prefer budgeted quality, label efficiency, stability, and runtime metrics.
- Keep recommendations bounded so a worker can run them locally.

## Execution Plan
- Inspect current learning-curve results -> verify which seeds/budgets fail.
- Inspect notebook dataset/model knobs -> verify which knobs are cheap to sweep.
- Produce hypotheses -> verify each maps to a concrete measurable experiment.
- Recommend matrix -> verify it is small enough for quick iteration.

## Acceptance Criteria
- Final report lists hypotheses, metrics, acceptance thresholds, and a small sweep matrix.
- Report explicitly says what would count as success or failure for each hypothesis.
- Report flags any risk that would make strategy comparison misleading.

## Expected Tests and Validations
- Read-only validation by citing inspected artifacts/values.

## Dependencies
- Depends on current learning-curve artifacts existing.

## Parallel/Sequential Notes
- Can run in parallel with the benchmark runner worker. Its output will be used to interpret/adjust subsequent rounds.
