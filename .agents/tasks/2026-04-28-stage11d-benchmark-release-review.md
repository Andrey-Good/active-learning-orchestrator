# Stage 11D Review: Real-Data Benchmark Release Fixes

## Task Identifier

stage11d-benchmark-release-review

## Context

Stage 11D implemented benchmark release fixes after Stage 11A/11B/11C rejected
readiness. This review must verify the integrated benchmark evidence contract,
not just test pass status.

## Goal

Determine whether Stage 11D closes all P1/P2 audit blockers without introducing
new release-quality problems.

## Responsibility Boundaries

In scope:

- `benchmarks/sdk_first_benchmark.py`
- `benchmarks/quality_gate_report.py`
- benchmark-related tests changed by Stage 11D
- `benchmarks/README.md`
- `docs/BENCHMARK_EVIDENCE.md`
- README benchmark wording
- Stage 11A/11B/11C audit reports

Out of scope:

- Editing files.
- Running long real dataset downloads.
- Production SDK runtime code.

## Files May Be Changed

- Only `.agents/tmp/2026-04-28-stage11d-benchmark-release-review.md`

## Review Questions

1. Are calibration metrics (`brier_score`, `nll`, `ece`) computed from aligned
   probabilities and included in budget and full-train artifacts?
2. Does quality gate summarize/render calibration metrics and fail promoted
   capped-real/standard evidence when they are missing?
3. Does the real standard contract enforce caps and at least three seeds without
   breaking quick smoke use cases?
4. Does evidence categorization now keep native external rows separate from
   formula comparison at top level and per strategy?
5. Do docs distinguish quick smoke, standard Stage 11, retained diagnostic, and
   command-support evidence honestly?
6. Are tests regression-oriented rather than only checking implementation details?
7. Did Stage 11D avoid production SDK runtime changes?

## Expected Validation

Run at least:

- `uv run pytest tests/test_quality_gate_report.py tests/test_sdk_first_benchmark_real_datasets.py tests/test_benchmark_evidence_contract.py -q`

Optional but useful:

- `uv run pytest -q`

## Acceptance Criteria

- Verdict must be accept or reject.
- Reject only for concrete P1/P2 blockers.
- If accepted, list residual P3 risks.

## Dependencies

- Stage 11A/11B/11C audits.
- Stage 11D worker patch.
