# W16 - Tie and Near-Tie Diagnostics

## Relation to Overall Task
Third benchmark-only diagnostic experiment after PCB rejection and temperature smoothing partial result. This checks whether uncertainty score ties or near-ties are a meaningful cause of unstable/skewed selection.

## Assumptions and Resolved Ambiguities
- W12 already fixed SDK deterministic tie-breaking.
- This task is benchmark-only and should not edit SDK source.
- If near-ties are low or jitter does not improve quality/skew, reject this line.

## Goal
Add tie/near-tie diagnostics and deterministic jitter variants for existing uncertainty heuristics.

## Responsibility Boundaries
- Own benchmark runner changes and generated tie/jitter artifacts only.

## In Scope
- Modify `benchmarks/run_learning_curve_experiments.py`.
- Add benchmark-only strategies:
  - `jitter_entropy_1e-6`;
  - `jitter_margin_1e-6`;
  - `jitter_least_confidence_1e-6`.
- Emit tie diagnostics:
  - `score_unique_fraction`;
  - `near_tie_rate_1e-12`;
  - `near_tie_rate_1e-6`;
  - selected overlap vs parent;
  - selected position metrics.
- Run on `tiny_transformer_unfrozen_fixed_dataset_default` with seeds `13,37,61`, budgets `16,32,48,64,80`, parents plus jitter variants and random.

## Out of Scope
- No SDK source changes.
- No temperature or PCB variants.
- No README changes.
- No varying confirmation unless a jitter variant passes quality criteria.

## Files/Modules May Change
- `benchmarks/run_learning_curve_experiments.py`
- New artifacts under `benchmarks/results/learning_curves/`, named with `tie_` or `jitter_`.

## Files/Areas Must Not Touch
- `src/active_learning_sdk/**`
- `README.md`
- Docker files

## Metrics and Success Criteria
Informative gate:
- near-tie rate `>= 25%` at `1e-6` or selected overlap vs parent `<= 0.95`.

Quality gate:
- AULC macro-F1 `>= parent + 0.020`, or skew reduction `>= 10%` with macro-F1@80 no worse than parent by more than `0.010`.

Promotion note:
- Jitter is not a promotion candidate unless it exposes a deterministic tie-order bug or robustly improves quality.

## Step -> Verify Plan
- Add score diagnostic helpers -> verify parent strategies can emit score stats.
- Add jitter variants -> verify deterministic selected IDs.
- Run fixed screen -> verify artifact integrity.
- Summarize informative/quality gates.

## Acceptance Criteria
- Artifacts clearly state whether ties/near-ties are a meaningful issue.
- Final report recommends continue/reject tie-jitter line.

## Expected Tests and Validations
- py_compile.
- tie/jitter command.
- Artifact integrity script.

## Dependencies
- Depends on R23 clean review.

## Parallel/Sequential Notes
- Must be independently reviewed after completion.
