# W14 - Temperature-Smoothed Uncertainty Sweep

## Relation to Overall Task
Second benchmark-only quality experiment after PCB was rejected. This tests whether uncertainty rankings are too sharp/poorly calibrated and can be improved by probability temperature smoothing before scoring.

## Assumptions and Resolved Ambiguities
- Do not edit SDK source.
- Temperature variants are benchmark-only.
- If temperature smoothing does not materially change selected samples, mark it uninformative rather than successful.

## Goal
Add and run a temperature-smoothed uncertainty sweep on the fixed dataset screen.

## Responsibility Boundaries
- Own benchmark runner changes and generated temperature artifacts only.

## In Scope
- Modify `benchmarks/run_learning_curve_experiments.py`.
- Add benchmark-only strategies:
  - `entropy_T2`, `entropy_T4`;
  - `margin_T2`;
  - `least_confidence_T2`.
- Semantics:
  - get probabilities from model;
  - transform `p_T = normalize((p + eps) ** (1/T))`;
  - score using parent heuristic on transformed probabilities;
  - select top-k by score with deterministic ordering consistent with existing notebook logic.
- Run on:
  - `tiny_transformer_unfrozen_fixed_dataset_default`;
  - seeds `13,37,61`;
  - budgets `16,32,48,64,80`;
  - strategies `random`, parents, and temperature variants.
- Compute selected overlap vs parent and mark whether each variant is informative.

## Out of Scope
- No SDK source changes.
- No PCB variants.
- No tie-jitter variants.
- No README changes.
- No varying-dataset confirmation unless a variant passes fixed-dataset success criteria.

## Files/Modules May Change
- `benchmarks/run_learning_curve_experiments.py`
- New artifacts under `benchmarks/results/learning_curves/`, named with `temperature_` or `temp_`.

## Files/Areas Must Not Touch
- `src/active_learning_sdk/**`
- `README.md`
- Docker files
- unrelated benchmark artifacts except if an existing command is explicitly smoke-run to temp output.

## Metrics and Success Criteria
Informative gate:
- selected overlap vs parent `<= 0.85` on average, or selected IDs differ in at least a meaningful fraction of rounds.

Quality gate:
- AULC macro-F1 improves by `>= 0.030` vs parent;
- cumulative label-delta drops by `>= 15%`;
- macro-F1@80 no worse than parent by more than `0.010`.

Promotion gate:
- also beats same-run random AULC macro-F1 by `>= 0.010` on fixed dataset, then confirm on varying dataset.

## Step -> Verify Plan
- Add temperature strategy resolver -> verify parent pure strategies unchanged.
- Run fixed screen -> verify rows/status/keys.
- Summarize overlap vs parent -> verify informative gate.
- Summarize lift/skew -> verify quality gate.

## Acceptance Criteria
- Artifacts clearly state supported/rejected/uninformative for each variant.
- Final report includes promotion recommendation.
- Existing `--help` parses and previous artifacts are not overwritten.

## Expected Tests and Validations
- `.venv\\Scripts\\python.exe -m py_compile benchmarks\\run_learning_curve_experiments.py`
- Temperature sweep command
- Artifact integrity script for row counts, unique keys, statuses, summary math, overlap/lift/skew.

## Dependencies
- Depends on R21 clean review of PCB sweep.

## Parallel/Sequential Notes
- Must be independently reviewed after completion.
