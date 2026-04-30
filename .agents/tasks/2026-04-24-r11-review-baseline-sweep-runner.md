# R11 - Review Baseline Sweep Runner

## Relation to Overall Task
Independent review of W06 baseline sweep runner and generated artifacts before we use those numbers to guide the next scientific iteration.

## Assumptions and Resolved Ambiguities
- W06 was allowed to change only benchmark automation, the learning-curve notebook, and learning-curve result artifacts.
- W06 reports that no candidate passed baseline acceptance.
- The reviewer must validate both code quality and whether the generated metrics actually support that conclusion.

## Goal
Find any defects, risks, missing validations, or misleading metrics in the baseline sweep implementation and artifacts.

## Responsibility Boundaries
- Read-only review.
- Do not edit files.

## In Scope
- Review `benchmarks/run_learning_curve_experiments.py` diff and behavior.
- Review `lab/learning_curve_lab.ipynb` relevant changes.
- Review `benchmarks/results/learning_curves/baseline_sweep_*` artifacts.
- Check deterministic seed-order behavior remains valid.
- Check acceptance criteria are encoded and summarized correctly.
- Check no unrelated areas were touched by W06.

## Out of Scope
- Do not propose new AL methods.
- Do not fix code.
- Do not rerun long benchmarks unless needed; small validation commands are OK.

## Files/Areas May Read
- `benchmarks/run_learning_curve_experiments.py`
- `lab/learning_curve_lab.ipynb`
- `benchmarks/results/learning_curves/*`
- Git diff/status

## Files/Areas Must Not Touch
- Entire repo is read-only for this review.

## Architectural Constraints and Forbidden Actions
- Treat reported negative results as acceptable if they are measured honestly.
- Block only for defects that make the benchmark misleading, non-reproducible, or hard to maintain.
- Label findings as defect, risk, required improvement request, optional follow-up, or question.

## Execution Plan
- Inspect W06 diff -> verify scope and code maintainability.
- Inspect artifacts -> verify row counts, uniqueness, statuses, and summary math.
- Run narrow validation if needed -> verify claimed commands still pass.
- Report blocking findings first.

## Acceptance Criteria
- If satisfied, explicitly state: no remaining in-scope defects, blocking risks, blocking questions, or required improvement requests.
- If not satisfied, provide concrete blocking findings with files/lines and exact fixes expected.

## Expected Tests and Validations
- At minimum, inspect artifacts with a short script or PowerShell command.
- Prefer running `python -m py_compile benchmarks/run_learning_curve_experiments.py`.
- Run `--check-seed-order` if feasible.

## Dependencies
- Depends on W06 changes being present.

## Parallel/Sequential Notes
- Must complete before using the sweep as accepted evidence.
