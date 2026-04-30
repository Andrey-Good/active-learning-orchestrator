# Task W103: Hard Audit Benchmarks And Documentation Claims

## Context

The user requested tests, a senior critique document, and benchmarks comparing this SDK against alternatives or hand-written equivalents. This subtask owns benchmark harness quality, evidence claims, docs/README accuracy, and comparison viability with external active-learning libraries.

## Goal

Determine whether benchmark evidence is reproducible, fair, and strong enough to support README/doc claims. Identify benchmark flaws, missing baselines, overclaims, and lightweight comparison paths.

## Responsibility Boundaries

May inspect:

- `benchmarks/**` except generated `benchmarks/results/**`
- `docs/**`
- `README.md`
- benchmark-related tests under `tests/`
- packaging metadata in `pyproject.toml`

May write only:

- `.agents/tmp/w103-benchmarks-docs-findings.md`

Do not edit source code, docs, tests, or benchmark scripts unless explicitly redirected by the orchestrator.

## In Scope

- Benchmark reproducibility and manifest quality
- Whether benchmark scripts compare SDK vs manual work and external libraries honestly
- README/docs claims that exceed evidence
- External library feasibility for `modAL` / `skactiveml` on Python 3.12
- Missing benchmark tests that would catch invalid comparisons

## Out Of Scope

- Runtime-state implementation details except where benchmarks depend on them
- Strategy math internals except where benchmark fairness depends on them

## Architectural Constraints

- Avoid persistent dependency changes.
- If checking external packages, use ephemeral tooling only and report failure modes.
- Work with the dirty worktree as-is.

## Forbidden Actions

- Do not run destructive commands.
- Do not install persistent dependencies into the project.
- Do not modify `uv.lock`, `pyproject.toml`, source files, docs, tests, or benchmark scripts.

## Execution Plan

1. Read benchmark scripts/tests and current docs.
2. Check what evidence can be reproduced quickly.
3. Evaluate external-library comparison feasibility.
4. Write findings to `.agents/tmp/w103-benchmarks-docs-findings.md` with severity, evidence, and repro guidance.

## Acceptance Criteria

- Benchmark/documentation findings cite exact scripts/docs and concrete claim gaps.
- External-library comparison notes are practical and do not invent results.
- Recommendations include specific benchmark/test artifacts the orchestrator can add.

## Dependencies

Can run in parallel with W101 and W102. No write-scope overlap.
