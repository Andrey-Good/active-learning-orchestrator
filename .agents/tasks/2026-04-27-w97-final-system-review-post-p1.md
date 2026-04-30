# Task W97-M: Final System Review After P1 Fixes

## Context
This is the final senior acceptance review after W97 audit/fix loops.

Previously found P1 blockers were fixed:
- reference benchmark no longer normalizes invalid probability rows;
- reference benchmark has no-clobber output handling and reproducibility metadata;
- simulator resume validates deterministic task IDs before recreating tasks;
- simulator restore now also rejects missing and unexpected persisted task IDs.

## Known Validation
- `uv run pytest tests/test_reference_strategy_benchmark.py tests/test_w97_runtime_state_backends.py tests/test_label_backends.py -q` -> `39 passed`
- `uv run pytest tests/test_w97_runtime_state_backends.py tests/test_label_backends.py -q` after extra edge cases -> `28 passed`
- `uv run pytest -q` -> `350 passed`
- `uv build` -> success, source distribution and wheel built
- `benchmarks/audit_sdk_vs_manual.py --output-dir benchmarks/results/w97_audit_sdk_vs_manual_post_review --repeats 5000` -> exact parity, max overhead `2.94x`
- `benchmarks/reference_strategy_benchmark.py --preset smoke --no-include-external --output-dir benchmarks/results/w97_reference_smoke_post_review` -> `66` metrics rows, `66` selection rows, `24` equivalence rows
- `benchmarks/sdk_first_benchmark.py --preset smoke --datasets separable_topics --strategies random,entropy --budgets 12 --seeds 13 --output-dir benchmarks/results/w97_sdk_smoke_post_review` -> `2` metrics rows, `6` stop-policy rows

## Goal
Perform a final read-only release-quality review. Decide whether any real senior-level release blockers remain.

## Scope
Read-only review of:
- SDK runtime/state/backend/cache/strategy code touched in W97;
- W97 tests;
- benchmark harness changes;
- README/benchmark docs honesty;
- packaging/dependency split.

Do not edit files.

## Acceptance Questions
- Are previous P0/P1 blockers fixed rather than papered over?
- Do tests assert real behavior, including edge cases?
- Are benchmark artifacts and docs honest and reproducible enough?
- Does packaging avoid unnecessary heavy core dependencies without breaking public imports?
- Does any remaining issue block considering this SDK release-candidate quality?

## Output
Return findings ordered by severity. If no release blockers remain, say so clearly and list residual risks.
