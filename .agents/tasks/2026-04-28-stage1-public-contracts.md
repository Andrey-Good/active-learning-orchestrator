# Stage 1: Public SDK Contracts

## Context

The release-hardening roadmap starts by freezing and documenting public contracts before further benchmarks, refactors, adapters, or release work. The SDK already exposes a broad root public API in `src/active_learning_sdk/__init__.py`, but there is no single product-facing contract document that defines stable/provisional/internal surfaces and behavioral guarantees.

## Goal

Create a clear public contract specification and tests that protect it from accidental drift.

## Ownership

You may edit:

- `docs/SDK_CONTRACTS.md`
- `docs/README.md`
- `src/active_learning_sdk/__init__.py` only if the documented public surface reveals a clear export mistake
- `tests/test_public_contracts.py` or a similarly named focused contract test file

Do not edit runtime behavior, strategies, backends, benchmark scripts, or adapters unless a public export needs a tiny correction.

## In Scope

- Define API stability tiers: stable, provisional, internal.
- Document public contracts for:
  - `ActiveLearningProject`
  - configuration dataclasses
  - dataset samples and annotation records
  - `SelectionContext`
  - strategies/scheduler
  - caches
  - label backends
  - model adapters/capabilities
  - reports/state/resume guarantees at a high level
  - exception categories
- Make error taxonomy explicit: user/configuration, backend, model adapter, infrastructure, state corruption, strategy, stop criteria.
- Add contract tests that assert:
  - root `__all__` contains documented stable exports;
  - stable exports can be imported from `active_learning_sdk`;
  - optional concrete adapters are not required for importing root package;
  - exceptions are importable and subclass the SDK base error.

## Out Of Scope

- New features.
- Benchmark evidence.
- Refactoring implementation modules.
- Rewriting README.
- Guaranteeing every internal module path as stable.

## Constraints

- Do not overpromise. If a surface is useful but still evolving, mark it provisional.
- Keep docs actionable and future-maintainer friendly, not marketing copy.
- Contract tests should be focused and cheap.
- Avoid dependency on optional packages in contract tests.

## Forbidden Actions

- Do not remove existing exports without orchestrator approval.
- Do not make optional dependencies mandatory.
- Do not weaken existing tests.
- Do not edit historical audit files except `docs/README.md` index links.

## Suggested Validation

- `uv run pytest tests/test_public_contracts.py -q`
- `uv run pytest -q`
- `uv run mypy src`
- `uv run --with ruff ruff check .`

## Acceptance Criteria

- `docs/SDK_CONTRACTS.md` exists and explains stable/provisional/internal boundaries.
- Tests pin the documented root public surface.
- Full test/static gates remain green.
