# Stage 11B: Quality Gate And Metrics Audit

## Task Identifier

stage11b-quality-gate-audit

## Context

`benchmarks/quality_gate_report.py` turns raw benchmark CSVs into claimable
evidence. If this report is too weak, benchmark numbers can mislead even when
the harness itself runs.

## Goal

Audit quality-gate logic and tests for Stage 11 real-data benchmark evidence.

## Responsibility Boundaries

In scope:

- quality gate report parsing and aggregation;
- required metric columns and optional metric handling;
- multi-seed mean/std behavior;
- evidence category detection;
- tests in `tests/test_quality_gate_report.py` and related benchmark contract
  tests.

Out of scope:

- Editing code.
- Running long benchmarks.

## Files May Be Changed

- Only `.agents/tmp/2026-04-28-stage11b-quality-gate-audit.md`

## Review Questions

1. Does quality gate enforce random baseline completeness?
2. Does it distinguish single-seed smoke from standard multi-seed reports?
3. Does it surface calibration metrics if present?
4. Does it fail when Stage 11-required metrics are absent from promoted real
   reports?
5. Does evidence categorization prevent formula/native conflation?
6. Are generated JSON and Markdown strict and useful?

## Expected Output

Write a report with accept/reject and concrete blockers.

## Forbidden Actions

- No code edits.
- No artifact edits.
