# 2026-04-27-w108 Senior benchmark/docs audit

## Context

The user asked for tests that expose real problems, a text file of all code review objections, and benchmarks showing where this SDK is worse than analogs or manual work. Optional comparison with `modAL` / `skactiveml` is allowed, but the repo must not be polluted.

## Goal

Audit benchmark validity, direct external-library comparison evidence, generated result hygiene, README/report claims, and packaging/documentation maintainability.

## Ownership

May read all repo files. May run benchmark scripts and install optional packages through ephemeral `uv run --with ...` commands. May propose or add audit output under `benchmarks/results/senior_acceptance_2026_04_27/` and final docs under `docs/` only through the orchestrator. Do not edit SDK implementation files.

## In Scope

- `benchmarks/`
- `docs/`
- `README.md`
- `pyproject.toml`
- benchmark-related tests

## Out Of Scope

- Fixing SDK runtime or strategy bugs
- Docker/Label Studio manual integration unless needed for a reproducible benchmark claim

## Constraints

- Do not vendor external libraries or write dependency files just to compare.
- Use `modAL-python` for the active-learning library distribution and `scikit-activeml` for skactiveml comparison if installing.
- Separate direct external-library API calls from benchmark-local formula shims.
- Report timing with enough context that results are not overclaimed.

## Execution Plan

1. Inspect benchmark scripts, outputs, and docs claims.
2. Run focused benchmark comparison commands if environment allows.
3. Identify evidence quality gaps and concrete benchmark results.

## Acceptance Criteria

- Benchmark conclusions include exact commands and output paths.
- Claims distinguish "SDK worse than manual formulas" from "SDK worse than external library".
- No dependency pollution or unrelated generated files.

## Dependencies

Can run in parallel with runtime/state and strategy/cache audits.
