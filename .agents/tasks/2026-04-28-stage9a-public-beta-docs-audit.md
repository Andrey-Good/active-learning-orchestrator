# Stage 9A: Public Beta Docs/Claims Audit

## Context

Stage 4 strategy-quality hardening is closed. The next roadmap gate is public beta coherence: package extras and install docs should be coherent, quickstarts should be smoke-testable, adapter limitations should be explicit, and public docs should not overclaim benchmark evidence.

## Goal

Perform a read-only audit of current public-facing docs and README claims. Identify concrete P1/P2 issues that block calling the repo coherent for controlled public beta.

## Read Scope

- `README.md`
- `docs/README.md`
- `docs/SDK_CONTRACTS.md`
- `docs/SDK_REAL_PRODUCT_ROADMAP.md`
- `docs/BENCHMARK_EVIDENCE.md`
- `benchmarks/README.md`
- `pyproject.toml`

## Write Scope

- `.agents/tmp/2026-04-28-stage9a-public-beta-docs-audit.md` only

## In Scope

- Install/extras documentation accuracy.
- Whether quickstarts are discoverable and match public APIs.
- Whether real adapter limitations are explicit.
- Whether benchmark claims are linked to artifacts and do not overclaim.
- Whether docs index points users to current authoritative documents.
- Stale test-count/status claims.

## Out Of Scope

- Do not edit docs or code.
- Do not audit production algorithms except where docs claim unsupported behavior.

## Acceptance Criteria

- Audit file exists with accept/reject and prioritized findings.
- Findings include exact files/sections and suggested fixes.
- If no P1/P2 issues, explicitly state residual non-blocking work.
