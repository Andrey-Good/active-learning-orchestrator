# Task W98-F: Review Public Split And Prelabel Contract Fixes

## Context
W98-C implemented column split resolution and prelabel min-confidence filtering.

## Goal
Read-only review: decide whether real issues remain in this scope.

## Scope
Read only:
- `src/active_learning_sdk/configs.py`
- `src/active_learning_sdk/engine.py`
- `tests/test_acceptance_public_contract_2026_04_27.py`
- relevant split/prelabel/public tests

Do not edit files.

## Known Validation
- Acceptance with `--runxfail` -> `16 passed`
- Focused public/runtime/backend set reported by worker -> passed
- Full suite -> `370 passed`

## Review Questions
- Is `SplitConfig(mode="column")` now real and deterministic, or at least safely rejected?
- Are missing/unknown split values rejected clearly?
- Is `PrelabelConfig.min_confidence` enforced without breaking accepted prelabel payloads?

## Output
Return findings ordered by severity. If no issues remain, say so clearly.
