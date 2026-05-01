# W10 - Fix Diagnostics Validation

## Relation to Overall Task
Fixes R15 blocking findings in acquisition diagnostics. Diagnostics must be robust enough to support scientific conclusions and future benchmark reruns.

## Assumptions and Resolved Ambiguities
- R15 found diagnostics are numerically correct for current artifacts but not adequately guarded against dataset drift or overlap-key drift.
- This is a validation hardening task, not a strategy/algorithm change.
- Do not change measured acquisition behavior.

## Goal
Make acquisition diagnostics fail loudly if:
- regenerated dataset labels/fingerprint do not match raw run `dataset_fingerprint`;
- overlap selected/cumulative keys are missing for any expected pair.

## Responsibility Boundaries
- Own only benchmark runner and regenerated acquisition diagnostics artifacts.
- Do not change SDK source, strategies, README, Docker, or baseline sweep measurements unless diagnostics regeneration requires artifact updates.

## In Scope
- Modify `benchmarks/run_learning_curve_experiments.py`.
- Regenerate `acquisition_*` diagnostics artifacts.
- Add validation logic for raw fingerprint vs regenerated payload fingerprint per candidate/dataset/dataset_seed group.
- Replace silent `.get(..., set())` overlap fallback with explicit key assertions.
- Add or strengthen integrity checks if useful.

## Out of Scope
- No new metrics unless directly needed for validation.
- No new experiments.
- No strategy behavior changes.

## Files/Modules May Change
- `benchmarks/run_learning_curve_experiments.py`
- `benchmarks/results/learning_curves/acquisition_*`

## Files/Areas Must Not Touch
- `src/active_learning_sdk/**`
- `README.md`
- Docker files
- unrelated artifacts

## Required Fixes
- For each diagnostic group used to rebuild labels, assert there is exactly one expected raw `dataset_fingerprint`.
- Assert regenerated payload fingerprint equals that raw fingerprint before using label maps.
- In overlap diagnostics, expected strategy keys must exist; missing keys should raise a clear `AssertionError`/`KeyError` naming the missing run key.

## Step -> Verify Plan
- Add fingerprint validation -> verify current diagnostics still pass.
- Add overlap key assertions -> verify current diagnostics still pass.
- Regenerate diagnostics -> verify artifacts match expected row counts and no false zero-overlap rows.
- Run py_compile and diagnostics CLI -> verify success.

## Acceptance Criteria
- R15 findings are resolved.
- Current diagnostics artifacts remain numerically equivalent except for any added metadata columns.
- Final report includes changed paths and validations.

## Expected Tests and Validations
- `.venv\\Scripts\\python.exe -m py_compile benchmarks\\run_learning_curve_experiments.py`
- `uv run python benchmarks\\run_learning_curve_experiments.py --diagnostics --diagnostics-runs-path benchmarks\\results\\learning_curves\\baseline_sweep_runs.csv`
- Artifact integrity script checking fingerprints and overlap key completeness.

## Dependencies
- Depends on R15 review findings.

## Parallel/Sequential Notes
- Must receive independent re-review after completion.
