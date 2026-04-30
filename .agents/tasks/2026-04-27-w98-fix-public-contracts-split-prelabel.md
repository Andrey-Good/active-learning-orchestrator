# Task W98-C: Public Contract For Column Splits And Prelabel Confidence

## Context
Stress review added strict xfail probes in `tests/test_acceptance_public_contract_2026_04_27.py`.

## Goal
Turn public-contract xfails green by making the SDK behavior match validated/public configuration.

## Ownership
May change:
- `src/active_learning_sdk/configs.py`
- `src/active_learning_sdk/engine.py`
- `tests/test_acceptance_public_contract_2026_04_27.py`
- narrowly related existing split/prelabel tests

Must not change:
- strategy probability code
- cache implementation
- benchmark harnesses

## Problems To Fix
1. `SplitConfig(mode="column")` validates but later reaches `NotImplementedError`.
2. `PrelabelConfig.min_confidence` is validated/persisted but ignored by `_make_prelabels()`.

## Expected Behavior
- Either implement column split mode or reject it at validation/configuration time with `ConfigurationError`.
- Prefer implementation if narrow and safe: use sample metadata/data split column from dataset provider samples, build train/val/test splits, and reject unknown split values or missing split fields.
- Enforce `min_confidence`: low-confidence prelabels must not be sent to the backend.
- If prelabels include confidence payloads, keep backend compatibility; if backend expects raw labels, preserve existing shape for accepted prelabels.

## Acceptance Criteria
- Focused public-contract tests pass without xfail.
- Existing split/prelabel/public API tests pass.
- No hidden scaffold `NotImplementedError` remains for validated public config.
