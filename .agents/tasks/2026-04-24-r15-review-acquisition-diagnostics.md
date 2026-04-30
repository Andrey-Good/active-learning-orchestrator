# R15 - Review Acquisition Diagnostics

## Relation to Overall Task
Independent review of W09 acquisition diagnostics. These diagnostics will drive the next optimization hypothesis, so they must be trustworthy.

## Assumptions and Resolved Ambiguities
- W09 should not have changed acquisition behavior or SDK source.
- Diagnostics are generated from `baseline_sweep_runs.csv` and selected IDs.
- The main claimed finding is class/label skew rather than duplicate selections or high overlap with random.

## Goal
Validate the diagnostics implementation, artifacts, and conclusions.

## Responsibility Boundaries
- Read-only review.
- Do not edit files.

## In Scope
- Review diagnostics code in `benchmarks/run_learning_curve_experiments.py`.
- Review generated `acquisition_*` artifacts.
- Recompute key checks from raw runs:
  - label counts sum to selected batch/cumulative sizes;
  - duplicate selected IDs are zero;
  - overlaps are bounded and paired correctly;
  - reported max skew and overlap claims match artifacts.
- Verify CLI behavior is documented/helpful enough for internal benchmark use.

## Out of Scope
- Do not implement fixes.
- Do not add or change strategies.
- Do not rerun expensive training; diagnostics-only commands are OK.

## Files/Areas May Read
- `benchmarks/run_learning_curve_experiments.py`
- `benchmarks/results/learning_curves/baseline_sweep_runs.csv`
- `benchmarks/results/learning_curves/acquisition_*`

## Files/Areas Must Not Touch
- Entire repo is read-only.

## Architectural Constraints and Forbidden Actions
- Block if diagnostics can silently produce misleading class skew or overlap metrics.
- Optional suggestions should be separated from blocking findings.

## Execution Plan
- Inspect code -> verify it uses selected IDs and payload labels consistently.
- Inspect artifacts -> verify schema and rows.
- Recompute key metrics -> compare to JSON/Markdown claims.
- Report findings.

## Acceptance Criteria
- If clean, explicitly state no remaining in-scope defects, blocking risks, blocking questions, or required improvement requests.
- If not clean, list concrete defects and expected fixes.

## Expected Tests and Validations
- `uv run python benchmarks/run_learning_curve_experiments.py --diagnostics --diagnostics-runs-path benchmarks/results/learning_curves/baseline_sweep_runs.csv` if feasible.
- Artifact recompute script.

## Dependencies
- Depends on W09 changes.

## Parallel/Sequential Notes
- Must complete before diagnostics are used to choose code changes.
