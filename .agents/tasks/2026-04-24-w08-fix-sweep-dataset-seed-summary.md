# W08 - Fix Sweep Dataset Seed Summary

## Relation to Overall Task
Fixes the blocking R13 review finding in the fixed-dataset/unfrozen baseline sweep. The issue is misleading summary metadata, not the raw measurements.

## Assumptions and Resolved Ambiguities
- R13 found that non-fixed candidates use multiple dataset seeds in raw rows but summary reports only `dataset_seed: 13`.
- Raw rows and acceptance math were otherwise validated.
- This fix must preserve W07 metrics and artifacts while making summary metadata honest.

## Goal
Update sweep summaries so dataset seed metadata is represented accurately:
- scalar only when all rows for a candidate share one dataset seed;
- otherwise list/count/variation marker for multiple dataset seeds.

## Responsibility Boundaries
- Own only benchmark runner and regenerated learning-curve artifacts.
- Do not change SDK source.
- Do not change README.

## In Scope
- Modify `benchmarks/run_learning_curve_experiments.py`.
- Regenerate `baseline_sweep_*` artifacts.
- If needed, refresh `learning_curve_*` only through the existing default validation command.
- Add/adjust artifact integrity validation for dataset seed summaries.

## Out of Scope
- No new candidates.
- No new strategies.
- No changes to acceptance thresholds.
- No broad refactors.

## Files/Modules May Change
- `benchmarks/run_learning_curve_experiments.py`
- `benchmarks/results/learning_curves/baseline_sweep_*`
- Existing `learning_curve_*` artifacts only if validation reruns default mode.

## Files/Areas Must Not Touch
- `src/active_learning_sdk/**`
- `README.md`
- Docker files
- unrelated benchmark scripts

## Required Fix
- Candidate summaries must include accurate fields, for example:
  - `dataset_seed_count`;
  - `dataset_seeds`;
  - `dataset_seed_mode` or similar (`fixed`/`varying`);
  - optional `dataset_seed` scalar only if unambiguous, or `NA`/empty when varying.
- Markdown/JSON top-candidate output must not imply a varying-seed candidate has a single fixed dataset seed.

## Step -> Verify Plan
- Update summary aggregation -> verify non-fixed candidates show multiple seeds and fixed candidates show one.
- Regenerate sweep artifacts -> verify no metric regressions except metadata columns.
- Run py_compile -> verify syntax.
- Run artifact integrity/recompute script -> verify candidate coverage, no duplicate keys, pass/fail unchanged, seed summary correctness.

## Acceptance Criteria
- R13 finding is resolved.
- `tiny_transformer_unfrozen_default` remains the only passing candidate unless metrics legitimately changed from rerun.
- Fixed-dataset candidates have one dataset seed/fingerprint; non-fixed candidates show multiple seeds/fingerprints.
- Final report includes validations and changed paths.

## Expected Tests and Validations
- `.venv\\Scripts\\python.exe -m py_compile benchmarks\\run_learning_curve_experiments.py`
- `.venv\\Scripts\\python.exe benchmarks\\run_learning_curve_experiments.py --sweep`
- Artifact integrity script checking dataset seed summary fields.

## Dependencies
- Depends on R13 review finding.

## Parallel/Sequential Notes
- Must be independently reviewed again after completion.
