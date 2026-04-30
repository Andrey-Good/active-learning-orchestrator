# R08A API/State Black-Box Review

Task identifier: `wave8-r08a-api-state-review`

## Goal

Independently review and, where practical, re-run the strongest W08A/W08B findings:

- W08A-001: custom backend `None` return payloads leak raw `AttributeError`.
- W08A-002: custom backend non-string task ids are accepted.
- W08B-001: `export_dataset_split(..., which="train"|"val"|"test")` rejects explicit configured split names.

## Ownership

May write only under `.agents/tmp/blackbox_stress_wave8/reviews/r08a_api_state/`.

Must not touch product files, docs, tests, benchmark source, worker directories, or another reviewer's directory.

## Black-Box Boundary

Do not inspect implementation source under `src/**`, repository tests under `tests/**`, or benchmark implementation files. You may read public docs/contracts, worker harnesses, worker results/logs, and generated artifacts.

## Review Questions

- Is the worker using the public API correctly?
- Does the documented contract actually require the expected behavior?
- Could the observation be caused by a bad harness, invalid setup, or unrealistic custom backend behavior?
- Is the severity appropriate?
- Are reproduction commands complete?
- Should any finding be rejected, downgraded, reframed as documentation ambiguity, or accepted?

## Plan

1. Read the relevant worker findings and results.
2. Re-run the minimal candidate cases or create a smaller independent reproducer using only public APIs.
3. Check public docs/contracts for the expectation.
4. Write `review.md` with accepted/rejected/downgraded decisions and evidence.
5. If re-run evidence is generated, save it under the owned review directory.

## Acceptance Criteria

The review must give a clear verdict for each candidate finding and explain the reasoning with evidence.
