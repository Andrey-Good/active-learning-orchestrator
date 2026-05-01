# Stage 10A: Adapter And Capability Contract Audit

## Task Identifier

stage10a-adapter-capability-audit

## Context

Stage 9 made the SDK coherent enough for controlled beta documentation. Stage 10
must decide whether adapter and capability contracts are strong enough for a
professional preview. This subtask is read-only and should identify concrete
P1/P2 defects before implementation.

## Goal

Audit the current model adapter and strategy capability surface and write a
precise findings report.

## Responsibility Boundaries

In scope:

- `src/active_learning_sdk/adapters/base.py`
- `src/active_learning_sdk/adapters/__init__.py`
- strategy `required_capabilities` declarations
- engine capability validation paths
- public docs that describe adapter/capability contracts
- existing tests around capability inspection and fail-fast behavior

Out of scope:

- Editing production code or tests.
- Changing benchmark evidence.
- Implementing fixes.

## Files May Be Read

- `src/active_learning_sdk/**`
- `tests/**`
- `README.md`
- `docs/SDK_CONTRACTS.md`
- `docs/SDK_REAL_PRODUCT_ROADMAP.md`

## Files May Be Changed

- Only `.agents/tmp/2026-04-28-stage10a-adapter-capability-audit.md`

## Files Must Not Be Touched

- Production source files.
- Public docs outside the tmp report.
- Tests.

## Architectural Constraints

- Root import must remain dependency-light.
- Optional adapters must remain lazily imported.
- Strategies must fail fast when strict capabilities are enabled.
- Non-strict capability mode may defer failures to selection, but must not silently
  produce invalid selections.

## Special Attention

- Placeholder methods decorated as unsupported must not count as capabilities.
- Protocol stubs must not count as real implementations.
- Capability error messages should name the strategy and missing capability.
- Mix, hybrid, bandit, custom strategies must validate all relevant arms.

## Forbidden Actions

- Do not hide defects by weakening tests.
- Do not mark uncertain findings as blockers.
- Do not modify SDK code.

## Execution Plan

1. Inspect capability inspection and validation code.
2. Compare built-in strategies against declared `required_capabilities`.
3. Check tests for missing negative cases and public-contract drift.
4. Write findings grouped by P1/P2/P3, with file/line evidence where possible.

## Acceptance Criteria

- Report states accept/reject for Stage 10 capability contract readiness.
- Every blocker has a reproducible reason.
- Non-blocking polish is separated from release blockers.

## Expected Validations

- Read-only audit; no test run required unless useful for confirming a finding.

## Dependencies

- None.

## Parallelism

Can run in parallel with Stage 10B and Stage 10C audits because it is read-only.
