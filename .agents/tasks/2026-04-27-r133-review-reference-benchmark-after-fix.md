# R133 Review Reference Benchmark After Fix

## Context
Reviewer R132 found a P2 issue: the external formula comparison in W100's benchmark was tautological because documented modAL/skactiveml formula selections called the manual helper. The benchmark has since been patched to compute documented formulas independently and to run optional modAL/skactiveml scorers when available via an ephemeral uv overlay.

## Goal
Review the benchmark after the P2 fix and confirm whether R132's finding is resolved.

## Responsibility Boundaries
Read-only review scope:
- `benchmarks/acceptance_reference_comparison_2026_04_27.py`
- `benchmarks/results/acceptance_reference_2026_04_27/**`
- `.agents/tmp/w100-benchmark-findings.md`
- `.agents/tmp/r132-review-reference-benchmark-acceptance-audit.md`

Owned write scope:
- `.agents/tmp/r133-review-reference-benchmark-after-fix.md`

Do not modify SDK code, benchmark code, tests, docs, pyproject, or lock files.

## Review Questions
- Does the benchmark still call `audit.manual_select()` for external documented formulas?
- Are optional modAL/skactiveml scorer rows actually produced when packages are importable?
- Are the claims and caveats honest after the fix?
- Are artifacts complete and consistent with the latest run?

## Acceptance Criteria
Write a concise review note with findings ordered by severity. If no issues remain, say so clearly and list residual risk.
