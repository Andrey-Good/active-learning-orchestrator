# R08B Packaging/Quality Black-Box Review

Task identifier: `wave8-r08b-packaging-quality-review`

## Goal

Independently review W08C/W08D claims:

- W08C-001: HF adapter optional-extra behavior is inconsistent with public contract wording.
- W08D DQL-1 through DQL-3: strategy underperformance/concentration observations are diagnostic quality limitations, not correctness defects.
- W08C/W08D positive claims around build, twine, isolated installs, README quickstart, and benchmark smoke.

## Ownership

May write only under `.agents/tmp/blackbox_stress_wave8/reviews/r08b_packaging_quality/`.

Must not touch product files, docs, tests, benchmark source, worker directories, or another reviewer's directory.

## Black-Box Boundary

Do not inspect implementation source under `src/**`, repository tests under `tests/**`, or benchmark implementation files. You may read public docs/contracts, worker harnesses, worker results/logs, generated artifacts, and package metadata.

## Review Questions

- Is W08C-001 a real current inconsistency, a benign design choice, or missing documentation?
- Are W08D's strategy observations supported by the metrics and framed with fair severity?
- Do any W08D observations deserve promotion to defect, or should they stay diagnostic?
- Are the packaging/benchmark positive claims supported by logs and generated artifacts?
- Are skipped heavy extras correctly treated as coverage gaps rather than defects?

## Plan

1. Read W08C/W08D findings, results, logs, and generated metric/summary artifacts.
2. Re-run only small checks if necessary; avoid heavy extras unless already available.
3. Write `review.md` with verdicts for each claim.
4. Save any independent evidence under the owned review directory.

## Acceptance Criteria

The review must clearly list accepted claims, rejected claims, severity changes, and any residual risk.
