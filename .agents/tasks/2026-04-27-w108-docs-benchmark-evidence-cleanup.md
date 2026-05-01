# Task W108: Docs/Benchmark Evidence Cleanup

## Context

The all-objections backlog says current docs and benchmark naming can overstate evidence. Some docs still report stale pass/xfail counts and some benchmark rows are formula shims named like external-library rows.

## Goal

Make docs/benchmark labels honest without changing runtime SDK behavior.

## Ownership

You may edit:

- `README.md`
- `docs/*.md`
- `benchmarks/reference_strategy_benchmark.py`
- benchmark tests if needed

Do not edit:

- SDK production source under `src/active_learning_sdk`
- current acceptance tests unrelated to benchmark docs

## Required Fixes

1. Release-facing docs must not claim clean `381 passed` while current xfails exist. If W106 fixes them before integration, update to the new truth.
2. Mark historical reports as archival where they contain stale counts.
3. Rename benchmark-local `modal_*`/`skactiveml_*` formula shim rows to names that make formula-shim semantics explicit, unless the harness actually calls real external APIs.
4. Keep the direct external scorer microbenchmark separate from formula parity evidence.

## Constraints

- Do not invent evidence.
- Prefer precise language over optimistic marketing language.
- Preserve benchmark artifacts format where possible; if row names change, update tests/docs accordingly.

## Acceptance Criteria

- No release-facing doc should hide active xfails.
- Benchmark naming should not imply direct external workflow comparison when only formula shims are used.
- Run focused benchmark/doc tests if applicable.
