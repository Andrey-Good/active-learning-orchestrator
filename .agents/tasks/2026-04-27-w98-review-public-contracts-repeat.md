# Task W98-G: Repeat Review Public Split And Prelabel Contract Fixes

## Context
The first public-contract review found two issues after W98-C:
- `PrelabelConfig(enable=True)` did not validate `predict_proba` during configure/attach;
- DataFrame top-level split columns were not exposed to column split resolution.

Both were fixed.

## Goal
Read-only repeat review: decide whether any real issues remain in this public-contract scope.

## Scope
Read only:
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/dataset/provider.py`
- `tests/test_acceptance_public_contract_2026_04_27.py`
- related public/capability tests if needed

Do not edit files.

## Known Validation
- `uv run pytest tests/test_acceptance_public_contract_2026_04_27.py tests/test_strategy_capabilities.py -q` -> `24 passed`
- `uv run pytest tests/test_acceptance_runtime_state_2026_04_27.py tests/test_acceptance_strategy_correctness_2026_04_27.py tests/test_acceptance_public_contract_2026_04_27.py -q --runxfail` -> `18 passed`
- `uv run pytest -q` -> `372 passed`
- `uv build` -> success

## Review Questions
- Does prelabel enable now require `predict_proba` before PUSH?
- Does this validation work for configure and attach_runtime?
- Does DataFrame column split use ordinary top-level split columns?
- Are missing/unknown split values still rejected clearly?
- Did exposing extra DataFrame columns create an obvious schema/fingerprint regression?

## Output
Return findings ordered by severity. If no issues remain, say so clearly.
