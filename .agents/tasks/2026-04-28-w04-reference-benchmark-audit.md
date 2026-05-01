# 2026-04-28-w04-reference-benchmark-audit

## Context

Part of a senior acceptance audit requested on 2026-04-28. The user explicitly asked for benchmarks showing where this SDK is worse than analogs or manual implementation, and allowed installing modAL / skactiveml if useful while avoiding repository noise.

## Goal

Audit existing benchmark methodology, run or add a reproducible benchmark comparison against manual implementation and, if available without heavy noise, skactiveml. Produce a benchmark script/result artifact plus findings.

## Responsibility Boundaries

Owner may change only:

- `benchmarks/deep_audit_reference_comparison_2026_04_28.py`
- `benchmarks/results/deep_audit_2026_04_28/**`
- `.agents/tmp/2026-04-28-w04-reference-benchmark-findings.md`

Owner must not change:

- `src/**`
- existing `tests/**`
- existing benchmark scripts
- dependency files unless explicitly impossible without them

## In Scope

- Existing benchmark harness validity.
- SDK vs manual selection/runtime overhead.
- SDK vs skactiveml where feasible.
- Evidence that benchmark claims are fair, deterministic, and not cherry-picked.

## Out of Scope

- Fixing SDK strategy implementations.
- Editing package dependencies.
- Long-running external dataset experiments.

## Constraints

- Keep runtime bounded.
- Avoid polluting global environment or lock files.
- Prefer the existing `.venv` and transient installs only if already available or necessary.
- If external library installation is not feasible, document the blocker and compare against a manual baseline.

## Execution Plan

1. Inspect existing benchmark scripts/results and dependency availability.
2. Determine whether `skactiveml` or `modAL` is available or can be installed cleanly without lockfile churn.
3. Add/run a bounded benchmark that compares comparable acquisition behavior and overhead.
4. Write findings about benchmark methodology, SDK gaps, and result interpretation.

## Acceptance Criteria

- Benchmark output is reproducible from a documented command.
- Results expose real weaknesses or state that no worse case was found under the bounded scenario.
- Findings do not overclaim beyond measured data.

## Validation

- Run the benchmark script.
- Validate generated result files are internally consistent.

## Dependencies

Can run in parallel with W01, W02, and W03.
