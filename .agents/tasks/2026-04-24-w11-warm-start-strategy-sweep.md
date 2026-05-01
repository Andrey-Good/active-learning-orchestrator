# W11 - Warm-Start Strategy Sweep

## Relation to Overall Task
First optimization experiment after accepted baseline and acquisition diagnostics. Diagnostics suggest uncertainty heuristics can select class-skewed early batches; this task tests whether a random warm-start before uncertainty improves quality and/or reduces skew using existing strategies only.

## Assumptions and Resolved Ambiguities
- Do not change SDK strategy behavior yet.
- Do not add BADGE, k-center, or any new acquisition method.
- This is a benchmark/lab experiment over existing strategies and schedules.
- Use accepted artifacts as context:
  - `tiny_transformer_unfrozen_default` passes random@48 but has high variance and near-saturation by 80 labels.
  - `tiny_transformer_unfrozen_fixed_dataset_default` is more stable but just misses macro-F1@48.
  - Diagnostics show class skew, especially for uncertainty-style strategies.

## Goal
Add and run a compact schedule sweep that compares pure strategies against warm-start schedules:
- `random_then_entropy`;
- `random_then_margin`;
- `random_then_least_confidence`;
- optionally `random2_then_*` if runtime remains bounded.

The sweep should quantify whether warm-start improves AULC/final metrics and reduces label skew compared with pure uncertainty and random.

## Responsibility Boundaries
- Own benchmark runner and generated warm-start artifacts only.
- Do not change SDK source or strategy algorithms.
- Do not change README.

## In Scope
- Modify `benchmarks/run_learning_curve_experiments.py`.
- Add schedule parsing/selection for benchmark-only composite strategies.
- Generate warm-start artifacts under `benchmarks/results/learning_curves/`, such as:
  - `warm_start_runs.csv`;
  - `warm_start_strategy_summary.csv`;
  - `warm_start_lift_vs_baselines.csv`;
  - `warm_start_summary.json`;
  - `warm_start_summary.md`;
  - optional diagnostics artifacts if cheap and reusable.
- Run on a bounded candidate set:
  - `tiny_transformer_unfrozen_default`;
  - `tiny_transformer_unfrozen_fixed_dataset_default`;
  - optionally one `bert_tiny_fixed_dataset_default` control if runtime is acceptable.
- Use budgets `16,32,48,64,80` and seeds `13,37,61`.

## Out of Scope
- No SDK source changes.
- No new acquisition score formula.
- No long real-dataset runs.
- No acceptance-threshold changes.

## Files/Modules May Change
- `benchmarks/run_learning_curve_experiments.py`
- New/updated warm-start artifacts under `benchmarks/results/learning_curves/`

## Files/Areas Must Not Touch
- `src/active_learning_sdk/**`
- `README.md`
- Docker files
- unrelated artifacts except diagnostics if explicitly regenerated for warm-start output.

## Architectural Constraints and Forbidden Actions
- Composite schedule names must be clearly marked benchmark-only.
- Do not call a warm-start schedule a new SDK strategy.
- Preserve existing default/sweep/diagnostics commands.
- Keep generated artifacts reviewable in size.
- Record negative results honestly.

## Step -> Verify Plan
- Add schedule support -> verify pure strategy behavior is unchanged.
- Add warm-start command/option -> verify it writes dedicated artifacts, not overwriting baseline sweep.
- Run warm-start sweep -> verify all rows ok or failure reasons explicit.
- Summarize quality -> verify lifts vs random and pure uncertainty are computed.
- If diagnostics are generated -> verify skew reductions are computed from selected IDs.

## Acceptance Criteria
- Warm-start artifacts exist and include per-budget metrics.
- Summary clearly states whether warm-start improved AULC/final macro-F1 versus random and pure uncertainty.
- Summary includes skew-related evidence or links to generated diagnostics.
- Final worker report includes changed paths, command/runtime, key metrics, supported/rejected hypotheses, and validations.

## Expected Tests and Validations
- `.venv\\Scripts\\python.exe -m py_compile benchmarks\\run_learning_curve_experiments.py`
- Warm-start sweep command
- Existing `--sweep` smoke or argument parse check if feasible
- Existing `--diagnostics` command if touched
- Artifact integrity script checking unique keys, row counts, status, and summary math.

## Dependencies
- Depends on R16 clean review of diagnostics validation.

## Parallel/Sequential Notes
- Must receive independent review after completion.
