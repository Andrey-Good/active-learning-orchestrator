# Review Stage 2: Benchmark Evidence

## Context

Stage 2A/B/C added benchmark evidence contracts, native external smoke entrypoint, quality gate schema v2, current smoke report, and docs updates. Full suite currently passes locally.

## Goal

Review only. Confirm Stage 2 benchmark evidence is accurate, non-overclaiming, tested, and integrated consistently across docs/scripts.

## In Scope

- `docs/BENCHMARK_EVIDENCE.md`
- `benchmarks/README.md`
- `README.md`
- `benchmarks/sdk_first_benchmark.py`
- `benchmarks/reference_strategy_benchmark.py`
- `benchmarks/native_external_benchmark.py`
- `benchmarks/quality_gate_report.py`
- `benchmarks/results/current_benchmark_report.md`
- tests added/modified for benchmark evidence/native external/quality gate

## Review Questions

- Are formula parity, SDK quality, native external smoke, and end-to-end project claims clearly separated?
- Do generated manifests contain sufficient reproducibility metadata?
- Does the native external benchmark actually call native APIs in tests and skip missing optional deps cleanly?
- Does the quality gate report avoid conflating formula shims and native external workflow evidence?
- Are README and benchmark docs internally consistent with current validation counts and commands?
- Are any claims still overstated relative to evidence?

## Out Of Scope

- Implementing large real external-library benchmark.
- SDK runtime or strategy changes.

## Constraints

- Review only, do not edit files.
- Provide concrete file/line findings.
- If accepted, explicitly say Stage 2 is accepted.

## Suggested Validation

- `uv run pytest tests/test_benchmark_evidence_contract.py tests/test_native_external_benchmark.py tests/test_quality_gate_report.py tests/test_reference_strategy_benchmark.py -q`
- `uv run pytest -q`
- `uv run mypy src`
- `uv run --with ruff ruff check .`

## Acceptance Criteria

- No P1/P2 findings.
- Stage 2 can be treated as complete before Stage 3 begins.
