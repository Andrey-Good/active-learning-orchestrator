# W96 Senior Audit Fixes

## Context
`docs/SENIOR_SDK_CODE_AUDIT_2026-04-27.md` introduced red acceptance tests for five confirmed blockers.

## Goal
Make the new audit tests pass through production fixes without weakening expectations or regressing the existing suite.

## In Scope
- Strict persisted state validation for sample statuses and round/task invariants.
- `project.validate()` detection for broken selected/task mappings.
- `LLMLabelBackend` preserving original `DataSample` payloads.
- `PredictionCache` collision-safe keys.
- JSON-safe project configuration for `SchedulerConfig(mode="custom", custom_selector=...)` while preserving live runtime callable behavior.

## Acceptance Criteria
- `tests/test_senior_audit_runtime_state_2026_04_27.py` passes.
- `tests/test_senior_audit_strategies_cache_2026_04_27.py` passes.
- `tests/test_senior_audit_public_api_2026_04_27.py` passes.
- Existing suite remains green.
- Build succeeds.

## Constraints
- Do not serialize Python callables into `state.json`.
- Do not remove custom selector support through the public project facade.
- Preserve backward compatibility where practical.
- Do not revert unrelated W94/W95 fixes.
