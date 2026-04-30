# Review Stage 4D: Strategy Diagnostics And Evidence Honesty

## Context

Stage 4D added strategy diagnostics to expose cold-start fallback/effective strategy in scheduler snapshots and updated benchmark/docs wording for stochastic/committee proxy evidence.

## Goal

Review Stage 4D for compatibility, diagnostic correctness, JSON-safety, and benchmark claim honesty.

## Read Scope

- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/strategies/uncertainty.py`
- `src/active_learning_sdk/strategies/badge.py`
- `benchmarks/sdk_first_benchmark.py`
- `benchmarks/README.md`
- `README.md`
- `tests/test_core_refactor_characterization.py`
- `tests/test_badge_strategy.py`
- `tests/test_benchmark_evidence_contract.py`
- `tests/test_sdk_first_benchmark_embedding_diagnostics.py`

## Write Scope

- `.agents/tmp/2026-04-28-stage4d-diagnostics-evidence-review.md` only

## Review Questions

- Are diagnostics scoped to one selection call and cleared before the next call?
- Do fallback snapshots expose useful information without changing non-fallback snapshots unnecessarily?
- Are diagnostics JSON-safe and free of oracle label leakage?
- Is BADGE/uncertainty fallback recording accurate for k-center vs random fallback?
- Do benchmark manifest/docs now clearly say stochastic/committee proxy rows are integration/diagnostic only?
- Are tests meaningful enough to prevent claim/snapshot regressions?

## Constraints

- Review only, do not edit.
- Flag any P1/P2 compatibility or claim-honesty issue.

## Suggested Validation

```powershell
uv run pytest tests\test_badge_strategy.py tests\test_class_balanced_entropy_strategy.py tests\test_group_diverse_strategy.py tests\test_core_refactor_characterization.py tests\test_benchmark_evidence_contract.py tests\test_sdk_first_benchmark_embedding_diagnostics.py -q
uv run pytest -q
uv run mypy src
uv run --with ruff ruff check .
```

## Acceptance Criteria

- Explicit accept/reject.
- If accepted, Stage 4D can be treated as complete.
