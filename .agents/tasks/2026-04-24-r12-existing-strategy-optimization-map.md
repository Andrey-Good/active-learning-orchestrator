# R12 - Existing Strategy Optimization Map

## Relation to Overall Task
This read-only research task prepares the next scientific cycles after baseline stabilization. The user asked to improve everything that currently affects active learning quality before adding new methods.

## Assumptions and Resolved Ambiguities
- Do not add new strategies in this task.
- Existing strategies include random, entropy, margin, least-confidence, and scheduler modes already present in the SDK/notebook.
- Baseline selection is still in progress, so this task should define what to test once a stable baseline is selected.

## Goal
Create a prioritized map of existing SDK/benchmark levers that can affect active-learning outcomes and define measurable hypotheses for each.

## Responsibility Boundaries
- Read-only.
- Do not edit files.

## In Scope
- Inspect SDK strategy/scheduler implementation and notebook `strategy_scores/select_batch`.
- Identify quality-affecting levers already present or implied by current code, such as:
  - uncertainty score definitions;
  - tie-breaking;
  - probability validation/calibration;
  - batch size/budget schedule;
  - scheduler single/mix/bandit modes;
  - stop criteria;
  - random seeding;
  - cache invalidation;
  - train/validation split effects.
- For each lever, define one or more concrete hypotheses and metrics.
- Prioritize by expected impact and risk.

## Out of Scope
- Do not propose large missing methods like BADGE as the immediate next implementation step, except as a future roadmap item.
- Do not change code.
- Do not run expensive benchmarks.

## Files/Areas May Read
- `src/active_learning_sdk/**`
- `active_learning_lab.ipynb`
- `benchmarks/run_learning_curve_experiments.py`
- Existing task docs and benchmark artifacts if useful.

## Files/Areas Must Not Touch
- Entire repo is read-only.

## Architectural Constraints and Forbidden Actions
- Keep recommendations experimentally testable.
- Distinguish SDK correctness defects from research/quality hypotheses.
- Favor small, reviewable next tasks.

## Execution Plan
- Inspect existing strategy/scheduler implementation -> verify current behavior.
- Inspect notebook acquisition implementation -> identify mismatch with SDK behavior.
- Build prioritized optimization map -> verify every item has metrics and expected direction.

## Acceptance Criteria
- Final report has a prioritized list of levers.
- Each lever has: hypothesis, metric, expected improvement, failure mode, and suggested next worker task.
- Report clearly separates blockers from optional future improvements.

## Expected Tests and Validations
- Read-only evidence with file references or behavior references.

## Dependencies
- Can run while W07 baseline sweep is in progress.

## Parallel/Sequential Notes
- Output will guide subsequent worker tasks after a baseline candidate is selected.
