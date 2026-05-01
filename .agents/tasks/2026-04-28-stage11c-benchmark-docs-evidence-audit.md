# Stage 11C: Benchmark Docs And Evidence Claims Audit

## Task Identifier

stage11c-benchmark-docs-evidence-audit

## Context

The README and benchmark docs contain many historic numbers and claim-scoped
statements. Stage 11 must keep those claims honest and machine-verifiable.

## Goal

Audit public benchmark documentation for overclaims, stale validation counts,
missing Stage 11 instructions, or contradictions.

## Responsibility Boundaries

In scope:

- `README.md`
- `benchmarks/README.md`
- `docs/BENCHMARK_EVIDENCE.md`
- `docs/SDK_REAL_PRODUCT_ROADMAP.md`
- current benchmark report references

Out of scope:

- Editing docs.
- Recomputing benchmark numbers.

## Files May Be Changed

- Only `.agents/tmp/2026-04-28-stage11c-benchmark-docs-evidence-audit.md`

## Review Questions

1. Do docs clearly say capped real-data evidence is diagnostic, not universal
   proof?
2. Are standard real benchmark commands shown with at least three seeds?
3. Are calibration/runtime/coverage/stop-policy metrics documented accurately?
4. Are stale local test counts or old Stage 4/9 validation claims framed as dated
   historical evidence?
5. Are native external library comparisons separated from formula shims?

## Expected Output

Write a severity-ranked docs/evidence audit report.

## Forbidden Actions

- No docs/code/test edits.
