# W07 - Fixed Dataset and Unfrozen Model Sweep

## Relation to Overall Task
Second scientific baseline iteration after W06. The reviewed W06 sweep showed that all current candidates fail baseline acceptance and random seed variance remains too high. This task tests the next hypotheses: dataset regeneration noise and model capacity/training dynamics.

## Assumptions and Resolved Ambiguities
- W06 runner and artifacts were independently reviewed with no blocking findings.
- Current failed candidates are valid evidence, not a runner bug.
- We still must not add new active-learning methods until existing behavior is measurable.
- Active learning comparisons require separate control of dataset seed and acquisition/model seed.

## Goal
Extend the baseline sweep to test:
- fixed synthetic dataset across acquisition seeds;
- non-frozen `tiny_transformer`;
- optionally fixed-dataset easy synthetic variants if they are cheap.

Generate updated sweep artifacts and report whether any candidate now passes baseline acceptance.

## Responsibility Boundaries
- Own only benchmark runner, learning-curve notebook if documentation/regeneration cell must change, and learning-curve artifacts.
- Do not change SDK source.
- Do not change README.

## In Scope
- Add candidate metadata for optional `dataset_seed`.
- Ensure row identity and summaries include enough information to distinguish fixed-dataset candidates.
- Add candidates such as:
  - `bert_tiny_fixed_dataset_default` using dataset seed `13` and acquisition seeds `13,37,61`;
  - `bert_tiny_fixed_dataset_easy_clean` using dataset seed `13`;
  - `tiny_transformer_unfrozen_default`;
  - if runtime is acceptable, `tiny_transformer_unfrozen_fixed_dataset_default`.
- Regenerate `baseline_sweep_*` artifacts.
- Preserve existing default non-sweep behavior.
- Preserve seed-order reproducibility check for default mode.

## Out of Scope
- No BADGE, k-center, or new strategy implementation.
- No Docker/Label Studio changes.
- No full Hugging Face real-dataset matrix unless the quick synthetic sweep clearly needs it and remains under time.

## Files/Modules May Change
- `benchmarks/run_learning_curve_experiments.py`
- `lab/learning_curve_lab.ipynb`
- `benchmarks/results/learning_curves/baseline_sweep_*`
- Existing `learning_curve_*` artifacts only if the default command is rerun as part of validation.

## Files/Areas Must Not Touch
- `src/active_learning_sdk/**`
- `README.md`
- Docker files
- unrelated benchmark scripts

## Architectural Constraints and Forbidden Actions
- Do not hide failed candidates; summarize them honestly.
- Do not loosen acceptance thresholds just to get a pass.
- If non-frozen `tiny_transformer` is too slow, record failure/skip explicitly rather than silently omitting it.
- Keep candidate definitions small and understandable.

## Step -> Verify Plan
- Add dataset-seed support to sweep candidates -> verify fixed-dataset rows share one dataset fingerprint across acquisition seeds.
- Add new candidates -> verify CLI `--sweep-candidates` includes them.
- Regenerate sweep -> verify row count and unique keys include `sweep_candidate`.
- Recompute summaries -> verify fixed-dataset std and random@48/random@80 metrics are present.
- Validate runner -> run py_compile and artifact integrity checks.

## Acceptance Criteria
- Sweep artifacts include fixed-dataset candidates and at least one `tiny_transformer` non-frozen candidate, unless a clear runtime/error reason is recorded.
- Summary includes dataset seed/fingerprint information where relevant.
- Worker final report states which hypothesis was supported or rejected.
- Existing reviewed behavior remains intact.

## Expected Tests and Validations
- `python -m py_compile benchmarks/run_learning_curve_experiments.py`
- `.venv\\Scripts\\python.exe benchmarks\\run_learning_curve_experiments.py --sweep`
- `.venv\\Scripts\\python.exe benchmarks\\run_learning_curve_experiments.py --check-seed-order`
- Artifact integrity script checking row counts, duplicate keys, statuses, candidate coverage, and fixed-dataset fingerprint behavior.
- Notebook JSON validation if modified.

## Dependencies
- Depends on R11 approval of W06.

## Parallel/Sequential Notes
- Must be reviewed independently after completion.
