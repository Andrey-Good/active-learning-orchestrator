# Stage 2A: Benchmark Evidence Schema And Claims

## Context

Stage 2 hardens benchmark evidence. Existing SDK-first and reference harnesses already emit manifests for new runs, but docs still mix retained legacy artifacts, formula parity, and production claims. We need a clear benchmark-evidence contract and tests that prevent claim drift.

## Goal

Create/strengthen benchmark evidence documentation and schema tests so every benchmark claim says exactly what it proves.

## Ownership

You may edit:

- `docs/BENCHMARK_EVIDENCE.md`
- `benchmarks/README.md`
- `README.md`
- `tests/test_benchmark_evidence_contract.py` or focused existing benchmark tests
- `benchmarks/sdk_first_benchmark.py` and `benchmarks/reference_strategy_benchmark.py` only for manifest/schema metadata fields, not algorithm behavior

Do not edit SDK runtime, strategies, adapters, or label backends.

## In Scope

- Define benchmark categories:
  - formula parity
  - SDK overhead
  - active-learning quality
  - native external-library workflow comparison
  - end-to-end public project workflow
- State which existing artifacts support each category.
- Mark retained Stage 9 artifacts as legacy/pre-schema where appropriate.
- Ensure newly generated SDK-first/reference manifests contain reproducibility metadata:
  - argv
  - git sha/dirty/status count
  - Python/runtime/platform
  - artifact schema version
  - artifact names
  - benchmark contract/claim category
- Add tests that run tiny benchmark smoke commands into temp dirs and assert manifest fields.

## Out Of Scope

- Implementing native external-library workflow benchmark itself; that is Stage 2B.
- Re-running large retained Stage 9 benchmarks.
- Editing old historical artifact JSON by hand unless strictly necessary and clearly documented.

## Constraints

- Do not overclaim real-world superiority from synthetic datasets.
- Do not call formula shims native external benchmarks.
- Keep tests fast.

## Suggested Validation

- `uv run pytest tests/test_benchmark_evidence_contract.py tests/test_reference_strategy_benchmark.py -q`
- `uv run pytest -q`
- `uv run --with ruff ruff check .`

## Acceptance Criteria

- Benchmark evidence docs make claims traceable and non-overstated.
- Manifest schema tests pass for new tiny runs.
- No full-suite regressions.
