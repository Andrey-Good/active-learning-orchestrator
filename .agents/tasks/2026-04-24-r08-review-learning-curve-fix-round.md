# Task ID: 2026-04-24-r08-review-learning-curve-fix-round

## Goal

Re-review the learning-curve runner after the seed-order and notebook regeneration fixes.

## Scope

Read-only review of:

- `benchmarks/run_learning_curve_experiments.py`
- `lab/learning_curve_lab.ipynb`
- `benchmarks/results/learning_curves/**`

## Specific Questions

- Is seed-order reproducibility fixed?
- Does `--check-seed-order` actually validate the right thing?
- Is the notebook regeneration cell executable as written?
- Are artifacts still honest about the failed random baseline?

## Acceptance Criteria

Return findings first. If no blocking findings remain, state that explicitly.
