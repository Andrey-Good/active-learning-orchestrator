# w92 - Audit Strategy Stress Tests

## Context

The SDK exposes many active-learning strategies. The user asked for a hard review that checks whether the code really works under edge cases and whether strategy behavior is defensible.

## Goal

Stress strategy selection logic with adversarial inputs and add focused tests that reveal correctness, determinism, leakage, invalid-output, and boundary-condition problems.

## Responsibility Boundaries

Own only:

- `tests/test_audit_strategy_edge_cases.py`
- read-only inspection of `src/active_learning_sdk/strategies/**`
- read-only inspection of `src/active_learning_sdk/configs.py`
- read-only inspection of `src/active_learning_sdk/types.py`

Do not edit SDK implementation files. Do not edit benchmark code. Do not edit runtime audit tests.

## In Scope

- Strategy behavior when requested batch size exceeds pool size.
- Duplicate selection prevention.
- NaN/Inf/negative probability handling.
- Probability rows that do not sum to one.
- Missing embeddings or wrong embedding shapes.
- Tie-breaking reproducibility.
- Class/group balancing under missing or adversarial metadata.
- Hybrid/mix scheduler edge cases where arms overlap or exhaust candidates.

## Out Of Scope

- Project lifecycle and state persistence.
- Label backend integration.
- Full benchmark runs.
- Fixing product code.

## Files/Areas Must Not Touch

- `src/**`
- `benchmarks/**`
- `tests/test_audit_runtime_edge_cases.py`

## Architectural Constraints

- Active-learning strategies must not use oracle labels for unlabeled pool selection.
- Selection output must be stable, unique, and restricted to the candidate pool.
- Tests must be deterministic and fast.

## Special Attention

- Look for strategies that silently accept malformed model outputs.
- Look for fallback behavior that returns plausible but invalid selections.
- Check whether different strategies share enough validation or duplicate it inconsistently.

## Forbidden Actions

- No destructive git operations.
- No dependency upgrades.
- No network downloads.
- Do not revert user changes.

## High-Level Plan

1. Inspect strategy APIs and current strategy tests.
2. Build small fake probability/embedding inputs that attack validation boundaries.
3. Add `tests/test_audit_strategy_edge_cases.py`.
4. Run the new test file and any narrow related strategy tests needed.
5. Return findings with code locations and changed paths.

## Acceptance Criteria

- New tests demonstrate real strategy robustness gaps or meaningful guardrail coverage.
- Findings are ordered by severity.
- The final response distinguishes algorithmic flaws from missing validation.

## Expected Tests And Validations

- `uv run pytest tests/test_audit_strategy_edge_cases.py -q`

## Dependencies

None.

## Parallel/Sequential Execution

Can run in parallel with runtime and benchmark audit tasks. Write scope is disjoint.
