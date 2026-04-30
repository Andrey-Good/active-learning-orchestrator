# Review W114-W117 Runtime/Cache/Data Fixes

## Context

Workers fixed current-open objections across runtime state loading, prediction cache scoping, strict DataFrame fingerprints, explicit split validation, JSONL cache performance/contract, and public API exports.

## Goal

Review only. Confirm correctness and identify regressions in:

- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/cache.py`
- `src/active_learning_sdk/dataset/fingerprint.py`
- `src/active_learning_sdk/dataset/provider.py`
- `src/active_learning_sdk/state/store.py`
- related tests

## In Scope

- Custom `StateStore` load semantics.
- Prediction cache dataset scoping and backwards safety.
- Strict fingerprint payload coverage.
- Explicit split coverage validation and legacy test updates.
- JSONL cache set/get performance contract and index correctness.
- Public API export stability if relevant.

## Out Of Scope

- Label Studio backend internals.
- Benchmark evidence wording except cache-related claims.
- Broad architecture refactors.

## Constraints

- Review only, do not edit.
- Provide concrete file/line findings.
- Severity P1/P2 only unless a P3 is very cheap and real.

## Suggested Validation

- `uv run pytest tests/test_current_open_audit_defects_2026_04_28.py tests/test_objection_sweep_state_cache_report_2026_04_28.py tests/test_acceptance_public_contract_2026_04_27.py tests/test_stop_criteria.py -q`
- Inspect `JsonlDiskCacheStore` for correctness after optimization.

## Acceptance Criteria

- State whether this scope is accepted.
- If rejected, list exact blockers and reproduction.
