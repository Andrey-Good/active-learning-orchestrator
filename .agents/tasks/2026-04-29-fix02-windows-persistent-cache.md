# Task: 2026-04-29-fix02-windows-persistent-cache

## Context

Black-box stress found that persistent prediction cache runs can fail on Windows with `PermissionError` while replacing `caches/predictions.index.json.tmp` with `caches/predictions.index.json`.

Evidence:

- `.agents/tmp/blackbox_stress/FINAL_STRESS_REPORT.md`
- `.agents/tmp/blackbox_stress/wave2_load_findings.md`
- `.agents/tmp/blackbox_stress/reviews/wave2_load_review.md`
- `.agents/tmp/blackbox_stress/wave2_load/raw_results.jsonl`

## Goal

Root-cause the Windows persistent cache replace failure and recommend a minimal safe fix and regression test.

## Responsibility Boundaries

Explorer scope:

- Inspect cache/state atomic-write code and black-box load artifacts.
- Identify why replacing the prediction cache index can raise `PermissionError`.
- Recommend focused tests that reproduce the issue without relying on huge benchmark runs.

Out of scope:

- Do not edit source files.
- Do not remove existing cache artifacts.
- Do not run destructive commands.

## Acceptance Criteria

- Write findings to `.agents/tmp/2026-04-29-fix02-windows-persistent-cache-findings.md`.
- Explain likely root cause and exact code area.
- Recommend a minimal fix and validation.
