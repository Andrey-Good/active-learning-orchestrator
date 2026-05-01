# w90 - Strict Capabilities Contract

## Context

The final review found that `SchedulerConfig.strict_capabilities` is exposed but not honored: capability validation always raises when a strategy requirement is missing.

## Goal

Make `strict_capabilities` behavior explicit and correct.

## Responsibility Boundaries

Own only:

- `src/active_learning_sdk/engine.py`
- config/API tests related to capability validation

Do not edit benchmarks, README, or strategy implementations.

## In Scope

- Inspect how `SchedulerConfig.strict_capabilities` is intended to work.
- If `strict_capabilities=True`, keep current hard failure on missing required model capabilities.
- If `strict_capabilities=False`, allow configuration/attachment to proceed while preserving runtime failure if the strategy is actually used without its required capability.
- Ensure custom strategies and hybrid/mix modes still validate sensibly.
- Add tests for both strict and non-strict behavior.

## Out Of Scope

- No changes to strategy `required_capabilities`.
- No benchmark or docs changes unless tests require small comments.

## Constraints

- Do not silently hide runtime errors when a missing capability is actually needed.
- Preserve default safety: strict validation should remain the default.
- Keep errors actionable and strategy-specific.

## Acceptance Criteria

- Tests cover strict default failure.
- Tests cover `strict_capabilities=False` configuration/attach acceptance.
- Runtime selection still fails clearly if the strategy needs a missing method.

## Tests

- Run the relevant capability tests plus full unit subset if practical.

## Dependencies

None.
