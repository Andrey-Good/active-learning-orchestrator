# 2026-04-28-w05-security-infra-backends-objection-sweep

## Context

Second-pass exhaustive senior audit requested on 2026-04-28. The user wants all defensible objections, not just a small sample.

## Goal

Audit security, infrastructure, process execution, Label Studio backend behavior, managed Docker behavior, HTTP handling, secrets, path handling, and backend idempotency. Produce additional failing tests only for concrete reproducible defects, plus a findings note.

## Responsibility Boundaries

Owner may change only:

- `tests/test_objection_sweep_security_infra_2026_04_28.py`
- `.agents/tmp/2026-04-28-w05-security-infra-backends-findings.md`

Owner must not change:

- `src/**`
- existing tests
- dependency files
- benchmark results
- docs except the owned findings file

## In Scope

- `backends/label_studio.py`
- `backends/managed_docker.py`
- `backends/base.py`
- `backends/simulator.py`
- Docker assets
- HTTP retry/error behavior
- label config escaping
- task/prediction parsing
- secret defaults and local runtime paths

## Out of Scope

- Strategy math.
- Public packaging.
- Engine state machine except backend-facing contracts.

## Constraints

- Do not require Docker or live Label Studio.
- Mock subprocess/network boundaries.
- Do not install dependencies or alter lockfiles.

## Acceptance Criteria

- Findings are evidence-backed, not speculative.
- Tests, if added, fail intentionally against current implementation and expose real behavior.
- Every objection includes severity and fix direction.

## Dependencies

Can run in parallel with W06, W07, W08, and W09.
