# 2026-04-28-w01-public-api-packaging-audit

## Context

Part of a senior acceptance audit requested on 2026-04-28. The goal is to find real maintainability and correctness defects, not to fix production code.

## Goal

Audit the public SDK contract, packaging metadata, optional dependency behavior, README claims, import boundaries, generated artifacts, and repository hygiene. Produce a focused failing/regression test file plus a concise findings note.

## Responsibility Boundaries

Owner may change only:

- `tests/test_deep_audit_public_api_packaging_2026_04_28.py`
- `.agents/tmp/2026-04-28-w01-public-api-packaging-findings.md`

Owner must not change:

- `src/**`
- existing `tests/**`
- `benchmarks/**`
- `README.md`
- `pyproject.toml`
- lock files

## In Scope

- Importability from a clean installed package perspective.
- Optional dependency isolation.
- Wheel/sdist package data correctness.
- README examples that do not match actual APIs.
- Public exports in `active_learning_sdk.__init__` and subpackages.
- Committed/generated noise such as cache files if it affects maintainability.

## Out of Scope

- Runtime engine behavior.
- Strategy math correctness.
- Benchmark quality comparisons.
- Fixing production implementation.

## Constraints

- Tests should be defensible and reproducible.
- Avoid tests that merely assert style preferences.
- Do not require network access.
- Do not install heavy new dependencies.

## Execution Plan

1. Read `pyproject.toml`, `README.md`, public package `__init__` files, and packaging-related tests.
2. Build or inspect package metadata if practical.
3. Add focused tests that expose concrete public contract or packaging defects.
4. Write findings with file/line evidence and reproduction commands.

## Acceptance Criteria

- The new test file contains real failure-oriented tests or regression assertions.
- Findings distinguish blockers from maintainability debt.
- Every finding has evidence and a concrete remediation direction.

## Validation

- Run the new test file directly.
- Run relevant existing packaging/public API tests if time permits.

## Dependencies

Can run in parallel with W02, W03, and W04.
