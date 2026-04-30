# Task W114: runtime state, prediction cache scope, DataFrame fingerprint, explicit split validation

## Context

The current open-defect sweep added strict xfail tests in `tests/test_current_open_audit_defects_2026_04_28.py`. Four failures belong to runtime/data consistency:

- injected `StateStore` is ignored unless `workdir/state.json` exists;
- `PredictionCache` is not scoped by dataset fingerprint;
- strict `DataFrameDatasetProvider` fingerprints ignore payload columns exposed to backends;
- explicit split maps can omit dataset ids at configure time.

## Goal

Make the four xfail tests pass under `--runxfail` without weakening existing state/caching/split contracts.

## Ownership

You may edit:

- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/cache.py`
- `src/active_learning_sdk/dataset/fingerprint.py`
- `src/active_learning_sdk/dataset/provider.py`
- `src/active_learning_sdk/state/store.py`
- relevant tests only if adding regression coverage is necessary

Do not edit Label Studio backend, benchmark scripts, docs, or sklearn adapter.

## Constraints

- Existing file-backed projects with missing `state.json` must still create a new state.
- Custom non-file `StateStore` should be loaded through its abstraction when possible.
- Prediction cache keys must remain backwards-safe: stale old entries may be missed, but must not be reused across dataset fingerprints.
- Strict fingerprints must cover the same JSON-safe payload that `DataFrameDatasetProvider.get_sample()` exposes to label backends.
- Explicit split validation should fail at `configure()` before saving invalid state.

## Forbidden Actions

- Do not remove tests or xfail markers.
- Do not bypass validation by disabling cache.
- Do not change unrelated strategies or benchmark code.

## Suggested Validation

- `uv run pytest tests\test_current_open_audit_defects_2026_04_28.py -q --runxfail`
- focused legacy tests touching cache/fingerprint/splits
- `uv run pytest -q`

## Acceptance Criteria

- The four runtime/data tests pass under `--runxfail`.
- Existing full suite remains green.
- Any new behavior is documented in code comments only where the contract is non-obvious.
