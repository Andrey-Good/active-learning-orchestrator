# 2026-04-29 W01 Public API Contract Stress

## Task Identifier

W01 - Public API, validation, and adversarial boundary stress.

## Context

Part of BLACKBOX-STRESS-MASTER. The SDK must be tested only through public documentation and public imports.

## Goal

Exercise documented root exports, configuration validation, dataset providers, model adapter contracts, scheduler modes, prelabeling, reports, exports, custom backend injection, and error taxonomy with adversarial but user-realistic inputs.

## Responsibility Boundaries

Write scope:
- `.agents/tmp/blackbox_stress_2026_04_29/w01_public_api/**`

Readable sources:
- `README.md`
- `docs/README.md`
- `docs/SDK_CONTRACTS.md`
- `docs/BENCHMARK_EVIDENCE.md`
- W01-generated artifacts

Must not read:
- `src/active_learning_sdk/**`
- `tests/**`
- benchmark implementation source

## In Scope

- Public root import checks.
- Bad model probability shapes, NaN/inf/negative/bool probability values, wrong label widths, stochastic/committee malformed cubes.
- Dataset edge cases: duplicate ids, missing text, non-string ids, very long text, empty pool, explicit split errors.
- Scheduler config edge cases for documented modes and strategy names.
- Custom backend and simulator use only if documented.
- Verify exception category where docs specify one.

## Out Of Scope

- SDK source fixes.
- Private state inspection beyond public APIs and generated project artifacts.
- Label Studio live service testing.

## Execution Plan

1. Build a standalone Python stress script using only public SDK imports.
2. Run many small isolated scenarios with fresh workdirs.
3. Record pass/fail/exception type/timing and minimal repro code fragments.
4. Produce `findings.md` and machine-readable `results.json`.

## Acceptance Criteria

- At least 40 adversarial scenarios are attempted.
- Findings are ranked P0/P1/P2/P3.
- Each P1/P2 finding has a minimal reproduction command.

## Parallelism

Can run in parallel with W02, W03, W04.
