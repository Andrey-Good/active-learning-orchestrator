# 2026-04-29 W04 Packaging Optional Integrations Stress

## Task Identifier

W04 - Packaging, extras, quickstart, and optional integration stress.

## Context

Part of BLACKBOX-STRESS-MASTER. The README documents install modes, extras, quickstarts, benchmark entrypoints, and optional adapters/backends.

## Goal

Find packaging/install issues, missing optional dependencies, root import pollution, quickstart breakage, docs/behavior drift, and optional integration failures without inspecting SDK source.

## Responsibility Boundaries

Write scope:
- `.agents/tmp/blackbox_stress_2026_04_29/w04_packaging/**`

Readable sources:
- `README.md`
- `docs/README.md`
- `benchmarks/README.md`
- `docs/LABEL_STUDIO_LIVE_TESTS.md`
- package metadata only as needed to run documented install commands
- W04-generated artifacts

Must not read:
- `src/active_learning_sdk/**`
- `tests/**`
- benchmark implementation source

## In Scope

- Fresh virtualenv install checks for core package and documented extras where feasible.
- Core simulator quickstart from README as a black-box script.
- Root import without optional extras.
- Optional sklearn adapter import/use if documented.
- Hugging Face adapter stance: verify documented scaffold behavior and failure clarity if dependencies are available or intentionally missing.
- Benchmark command smoke where documented and bounded.

## Out Of Scope

- Publishing, pushing, or modifying package metadata.
- Docker/Label Studio destructive operations unless explicitly isolated and bounded.
- Reading adapter source code.

## Execution Plan

1. Create isolated venvs under W04 output directory.
2. Install core editable package and optionally selected extras.
3. Run README quickstart and import probes.
4. Try documented benchmark smoke commands if dependencies are present or installable.
5. Produce `findings.md`, `install_logs/`, and `results.json`.

## Acceptance Criteria

- At least core install + quickstart are exercised.
- Optional extras are either tested or skipped with concrete environment reason.
- Findings distinguish environment failures from SDK packaging defects.

## Parallelism

Can run in parallel with W01, W02, W03.
