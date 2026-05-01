# Review Stage 1: Public SDK Contracts

## Context

Stage 1 added `docs/SDK_CONTRACTS.md`, updated `docs/README.md`, and added public contract tests. The goal was to freeze public contracts before benchmark/refactor work.

## Goal

Review only. Confirm that the contract document is accurate, not overpromising, and protected by useful tests.

## In Scope

- `docs/SDK_CONTRACTS.md`
- `docs/README.md`
- `tests/test_public_contracts.py`
- `src/active_learning_sdk/__init__.py` only for consistency with the documented contract

## Review Questions

- Does the document clearly separate stable, provisional, and internal surfaces?
- Does it accurately describe root exports and error taxonomy?
- Does it avoid promising unsupported behavior?
- Do tests pin meaningful contracts without making optional dependencies mandatory?
- Is anything important missing from the Stage 1 acceptance criteria?

## Out Of Scope

- Benchmarks.
- Runtime behavior changes.
- Large docs rewrite beyond contract correctness.

## Constraints

- Review only, do not edit files.
- Provide concrete findings with file/line references.
- If accepted, explicitly say Stage 1 is accepted.

## Suggested Validation

- `uv run pytest tests/test_public_contracts.py -q`
- optionally inspect `active_learning_sdk.__all__`

## Acceptance Criteria

- No P1/P2 findings.
- Stage 1 can be treated as complete before Stage 2 begins.
