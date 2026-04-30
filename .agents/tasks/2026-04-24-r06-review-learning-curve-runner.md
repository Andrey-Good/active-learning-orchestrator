# Task ID: 2026-04-24-r06-review-learning-curve-runner

## Relation To Overall Task

Independent review of the first learning-curve experiment surface.

## Goal

Verify that the new learning-curve runner/notebook/artifacts are scientifically useful, budget-matched, and honest about the failed random-baseline acceptance.

## Responsibility Boundaries

Read-only review. Do not edit files.

## In Scope

- `benchmarks/run_learning_curve_experiments.py`
- `lab/learning_curve_lab.ipynb`
- `benchmarks/results/learning_curves/**`

## Out Of Scope

- SDK engine changes;
- README edits;
- implementing new strategies.

## Files That May Be Changed

None.

## Files That Must Not Be Touched

The entire repository. Review only.

## Review Checklist

- Does the runner compare strategies at equal budgets?
- Are seeds handled cleanly?
- Are results one row per strategy/seed/budget?
- Does it report baseline acceptance failure clearly?
- Is the generated notebook useful and not a stale copy?
- Are artifacts reproducible and schema useful?
- Are there misleading claims in summary artifacts?

## Acceptance Criteria

- Return findings first, ordered by severity.
- If no blocking findings remain, explicitly state that.
