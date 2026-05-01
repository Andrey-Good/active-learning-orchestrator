# R13 - Review Fixed Dataset and Unfrozen Sweep

## Relation to Overall Task
Independent review of W07. We are about to use W07 results to choose a baseline and start acquisition diagnostics, so the sweep and conclusions must be trustworthy.

## Assumptions and Resolved Ambiguities
- W07 was allowed to change only benchmark runner, learning-curve notebook/artifacts if needed, and learning-curve result artifacts.
- W07 reports one passing candidate: `tiny_transformer_unfrozen_default`.
- Fixed-dataset fingerprint behavior is a key claim and must be checked.

## Goal
Review implementation, artifacts, and conclusions for correctness and misleading-result risks.

## Responsibility Boundaries
- Read-only review.
- Do not edit files.

## In Scope
- Review `benchmarks/run_learning_curve_experiments.py`.
- Review `benchmarks/results/learning_curves/baseline_sweep_*`.
- Check whether `dataset_seed`/fixed-dataset semantics are implemented correctly.
- Check whether candidate summary and passing-candidate logic are correct.
- Check whether the conclusion about saturation risk for `tiny_transformer_unfrozen_default` is supported.
- Run narrow validation commands if feasible.

## Out of Scope
- Do not implement fixes.
- Do not add new experiments or methods.
- Do not review unrelated dirty worktree files except to confirm W07 did not obviously touch them.

## Files/Areas May Read
- `benchmarks/run_learning_curve_experiments.py`
- `benchmarks/results/learning_curves/*`
- `lab/learning_curve_lab.ipynb` if it changed
- Git diff/status

## Files/Areas Must Not Touch
- Entire repo is read-only.

## Architectural Constraints and Forbidden Actions
- Negative results are acceptable if measured honestly.
- Do not require perfect benchmark design; block only defects that make this sweep misleading or non-reproducible.
- Label findings as defect, risk, required improvement request, optional follow-up, or question.

## Execution Plan
- Inspect diff and code paths -> verify dataset_seed and candidates.
- Inspect raw rows -> verify row counts, unique keys, statuses, candidate coverage, fixed fingerprints.
- Recompute critical summary values from raw rows -> verify reported pass/fail and top candidate.
- Report blocking findings first.

## Acceptance Criteria
- If satisfied, explicitly state no remaining in-scope defects, blocking risks, blocking questions, or required improvement requests.
- If not satisfied, provide exact blocking findings and expected fixes.

## Expected Tests and Validations
- At minimum, run a small artifact integrity/recompute script.
- Prefer `python -m py_compile benchmarks/run_learning_curve_experiments.py`.
- Do not rerun the full sweep unless necessary.

## Dependencies
- Depends on W07 changes being present.

## Parallel/Sequential Notes
- Must complete before baseline selection is considered accepted evidence.
