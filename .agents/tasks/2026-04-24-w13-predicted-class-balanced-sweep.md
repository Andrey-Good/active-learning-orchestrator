# W13 - Predicted-Class-Balanced Uncertainty Sweep

## Relation to Overall Task
Benchmark-only quality experiment based on accepted diagnostics and R19 recommendations. Current uncertainty often underperforms random and shows class-skewed selections. This task tests whether balancing selections by model-predicted class improves quality and skew without adding a public SDK method yet.

## Assumptions and Resolved Ambiguities
- W12 SDK correctness changes are accepted.
- Warm-start was rejected as a strict quality improvement.
- This task must not edit SDK source.
- `pcb_*` variants are benchmark-only strategy names.

## Goal
Add and run a compact predicted-class-balanced uncertainty sweep.

## Responsibility Boundaries
- Own benchmark runner changes and generated `pcb_*` artifacts only.

## In Scope
- Modify `benchmarks/run_learning_curve_experiments.py`.
- Add benchmark-only strategies:
  - `pcb_entropy`;
  - `pcb_margin`;
  - `pcb_least_confidence`.
- Strategy semantics:
  - score pool with parent uncertainty score;
  - compute predicted class via `argmax(probabilities)`;
  - allocate near-uniform quotas across predicted classes for the batch;
  - select highest parent-score items within each predicted class;
  - fill unfilled slots globally by parent score without duplicates.
- Run screen on:
  - `tiny_transformer_unfrozen_fixed_dataset_default`;
  - seeds `13,37,61`;
  - budgets `16,32,48,64,80`;
  - strategies `random`, `entropy`, `margin`, `least_confidence`, `pcb_entropy`, `pcb_margin`, `pcb_least_confidence`.
- If a variant passes success criteria on fixed dataset, confirm it on `tiny_transformer_unfrozen_default` with random, parent, and winning variants.

## Out of Scope
- No SDK source changes.
- No README changes.
- No real-dataset long matrix.
- No temperature smoothing in this task.
- No tie-jitter diagnostics in this task.

## Files/Modules May Change
- `benchmarks/run_learning_curve_experiments.py`
- New artifacts under `benchmarks/results/learning_curves/`, named with `pcb_` or `balanced_`.

## Files/Areas Must Not Touch
- `src/active_learning_sdk/**`
- `README.md`
- Docker files
- unrelated benchmark artifacts unless validation explicitly reruns an existing command.

## Metrics and Success Criteria
Primary:
- `aulc_macro_f1`;
- `macro_f1_at_48`;
- `macro_f1_at_80`.

Secondary:
- `aulc_accuracy`;
- `mean_cumulative_max_abs_label_delta_vs_pool`;
- `mean_selected_majority_label_fraction`;
- seed std for `macro_f1_at_48`.

Fixed-dataset pass:
- variant improves same-run parent AULC macro-F1 by `>= 0.050`;
- skew drops by `>= 0.050` absolute or `>= 25%`;
- macro-F1@80 is no worse than parent by more than `0.010`.

SDK promotion candidate:
- also beats same-run random AULC macro-F1 by `>= 0.010` on both fixed and varying confirmation candidates.

## Step -> Verify Plan
- Add benchmark-only selection helper -> verify pure strategy behavior remains unchanged.
- Add PCB runner command/artifacts -> verify no overwrite of baseline/warm-start artifacts.
- Run fixed screen -> verify artifact row counts/statuses.
- If fixed pass exists, run varying confirmation -> verify promotion criteria.
- Generate diagnostics/lift summaries -> verify summary math.

## Acceptance Criteria
- Artifacts clearly show supported/rejected hypothesis.
- Final report identifies whether any `pcb_*` should be promoted to SDK.
- Existing `--sweep`, `--warm-start`, and `--diagnostics` commands still parse or smoke-pass.

## Expected Tests and Validations
- `.venv\\Scripts\\python.exe -m py_compile benchmarks\\run_learning_curve_experiments.py`
- PCB sweep command
- Artifact integrity script for row counts, unique keys, statuses, summary math, skew deltas.
- Existing command smoke for `--help` and preferably diagnostics over generated PCB rows.

## Dependencies
- Depends on W12 review and R19 recommendations.

## Parallel/Sequential Notes
- Must be independently reviewed after completion.
