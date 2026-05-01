# W07D - Backend, Report, Export, Timeout, And Ops Stress

## Task Identifier

W07D-BACKEND-REPORT-OPS

## Context

The docs promise backend lifecycle contracts, deterministic simulator behavior, stop criteria, annotation timeouts, reports, exports, events, manifests, hashes, and project locking. This subtask attacks those operational surfaces through public APIs.

## Goal

Find reproducible backend/report/export/timeout/state-operation defects and weak error taxonomy.

## Responsibility Boundaries

May write only:
- `.agents/tmp/blackbox_stress_wave7/w07d_backend_report/**`

Must not touch:
- `src/active_learning_sdk/**`
- `tests/**`
- `benchmarks/*.py`
- existing docs, package metadata, lockfiles, or benchmark result dirs.

## In Scope

- Custom label backend protocol behavior through public contracts.
- Simulator backend black-box behavior.
- Annotation timeout policies and stop criteria.
- Report generation artifacts, manifest/hash presence, event/audit artifact presence.
- Export labels and dataset splits.
- Project lock/contention behavior using public project instances/processes.

## Out Of Scope

- Live Label Studio unless a local service is already trivially available.
- Managed Docker startup unless bounded and clearly recorded.
- Real dataset metric quality.

## Architectural Constraints

Use public backend protocol and facade methods only. Avoid private state mutation except deliberately corrupting files in a copied workdir to test documented state-corruption taxonomy.

## Special Attention

Do not confuse a custom backend violating the contract with an SDK defect unless the SDK mishandles taxonomy, state recovery, idempotency, timeout, or documented lifecycle order.

## Execution Plan

1. Read README backend/report/stop sections and docs/SDK_CONTRACTS.md.
2. Build standalone public API harness under the task directory.
3. Exercise backend lifecycle, timeout variants, report/export artifacts, project lock contention, and corrupt-state recovery.
4. Write `results.json`, logs, and `findings.md`.

## Acceptance Criteria

- At least 25 operational cases attempted.
- Report/export artifacts are checked by reading generated files, not SDK internals.
- Findings include command, observed behavior, expected behavior, severity, and reproduction notes.

## Dependencies

Can run in parallel with W07A/W07B/W07C.
