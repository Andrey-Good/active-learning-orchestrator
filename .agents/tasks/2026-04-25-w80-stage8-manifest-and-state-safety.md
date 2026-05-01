# W80 - Stage 8 Manifest And State Safety

## Context

Stage 8 report generator is complete. Remaining auditability gaps:

- `state.json` is written with default `json.dumps(...)`, so non-finite floats could leak if a model/backend returns them.
- State version handling only rejects unknown versions; there is no migration hook/manifest for future evolution.
- Reports contain useful state summaries but no standalone run manifest capturing SDK/package/runtime/config fingerprints.

## Goal

Add strict state serialization safety and a reproducibility manifest artifact.

## Responsibility Boundaries

You may change:

- `src/active_learning_sdk/state/store.py`
- `src/active_learning_sdk/report.py`
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/project.py` only for public API plumbing if needed
- tests, preferably `tests/test_state_safety.py` and/or extending `tests/test_report_generation.py`

Do not edit benchmark harness or README in this subtask.

## In Scope

- Make state save strict-JSON-safe:
  - sanitize or reject non-finite float values before writing state;
  - use `allow_nan=False`;
  - preserve deterministic ordering where practical.
- Add a small migration/compatibility hook:
  - centralize supported/current state version metadata;
  - make unsupported version errors explicit and test-covered;
  - do not invent complex migrations unless needed.
- Add a reproducibility manifest in report output:
  - SDK/package version if available;
  - Python version/platform;
  - state version;
  - project name;
  - dataset fingerprint/config summary;
  - scheduler/annotation/backend/cache configs;
  - artifact schema version for reports;
  - counts of rounds/metrics/samples;
  - generated artifact filenames.
- Ensure manifest is strict JSON and included in Markdown/HTML summary or linked/listed.
- Add tests:
  - state store rejects/sanitizes NaN/Infinity and writes strict JSON;
  - unsupported state version error is clear;
  - report manifest exists and includes reproducibility fields;
  - public `generate_report` writes manifest alongside summary/report artifacts.

## Out Of Scope

- No full historical migration framework.
- No README docs yet.
- No benchmark report aggregation yet.

## Architectural Constraints

- Do not mutate state during report generation.
- Prefer sanitizing report artifacts, but state persistence should be stricter: either sanitize non-finite values deterministically or fail with a clear error before writing invalid JSON.
- Keep stdlib-only.

## Acceptance Criteria

- `state.json` cannot contain `NaN`/`Infinity`.
- Report output includes a manifest artifact.
- Tests and full suite pass.

## Validation

- `uv run --group dev pytest -q tests/test_state_safety.py tests/test_report_generation.py`
- `uv run --group dev pytest -q`
