# Task: blackbox-stress-wave2-packaging-docs

## Context
Second wave of the user's aggressive black-box SDK stress test. Source-code inspection is forbidden; use documentation and public package behavior only.

## Goal
Stress installability, packaging, root exports, optional extras boundaries, wheel/sdist behavior, and documentation snippets from the perspective of a fresh external user.

## Responsibility Boundaries
Owns only artifacts under `.agents/tmp/blackbox_stress/wave2_packaging/` and findings in `.agents/tmp/blackbox_stress/wave2_packaging_findings.md`.

## In Scope
- Read public docs: `README.md`, `docs/README.md`, `docs/SDK_CONTRACTS.md`, `pyproject.toml`, and benchmark docs if needed.
- Build or use the local package as a consumer would.
- Create isolated temp virtual environments or `uv` run contexts under the owned directory.
- Test core install without optional extras, optional import boundaries, wheel contents, root exports, README quickstart/doc snippets, and packaging metadata.
- Verify that concrete optional adapters do not break root import when extras are absent.

## Out of Scope
- Reading `src/active_learning_sdk/**`.
- Modifying SDK source, docs, tests, or benchmarks.
- Fixing bugs.

## Must Not Touch
- `src/**`
- `tests/**`
- `benchmarks/**` except running documented public commands if needed
- other `.agents/tmp/blackbox_stress/**` directories

## Execution Plan
1. Build/install package into an isolated location or temp venv.
2. Run core quickstart from docs.
3. Probe optional extra behavior and packaging metadata.
4. Record failures, doc mismatches, and skipped checks with reasons.

## Acceptance Criteria
- At least 12 packaging/docs/import cases attempted.
- Findings include reproduction commands, environment description, and expected vs observed behavior.
- No SDK source inspection.
