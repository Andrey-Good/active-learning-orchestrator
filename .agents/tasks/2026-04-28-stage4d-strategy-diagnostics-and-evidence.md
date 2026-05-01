# Stage 4D: Strategy Diagnostics And Benchmark Evidence Honesty

## Context

Stage 4A found two remaining strategy-quality evidence gaps:

- Built-in uncertainty/BADGE strategies can internally switch to cold-start exploration, but single-strategy scheduler snapshots still say only `strategy=entropy`/`badge`.
- Stochastic/committee benchmark rows use deterministic proxy predictions from one sklearn model; they verify SDK integration, not true MC-dropout or independently trained committee quality.

Stage 4B/4C strategy formula fixes are accepted.

## Goal

Make fallback/exploration behavior visible in scheduler snapshots and ensure benchmark/docs clearly label stochastic/committee proxy rows as integration evidence.

## Ownership

May edit:

- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/strategies/base.py`
- `src/active_learning_sdk/strategies/uncertainty.py`
- `src/active_learning_sdk/strategies/badge.py`
- `benchmarks/sdk_first_benchmark.py`
- `benchmarks/README.md`
- `README.md`
- tests that directly cover scheduler snapshots or benchmark manifest wording

Must not edit:

- `src/active_learning_sdk/strategies/embedding.py`
- `src/active_learning_sdk/strategies/hybrid.py`
- `src/active_learning_sdk/strategies/stochastic.py`
- Label Studio/backend/state/cache files

## In Scope

- Add a small internal strategy-diagnostic recording mechanism on `SelectionContext`.
- Uncertainty and BADGE cold-start exploration should record diagnostics when they actually use fallback/exploration.
- Single-strategy scheduler snapshots should include fields such as `effective_strategy`, `fallback_reason`, support count/fraction, label count, and missing label count when a fallback occurred.
- Non-fallback snapshots should remain unchanged where practical.
- Benchmark manifest/docs should say stochastic/committee proxy strategies are integration/diagnostic evidence only, not true MC-dropout/independent committee quality evidence.
- Add focused tests proving fallback snapshots and benchmark wording.

## Out Of Scope

- Do not redesign public `SamplingStrategy.select`.
- Do not change strategy return types.
- Do not implement real MC-dropout or independently trained committee benchmark in this stage.
- Do not regenerate large benchmark artifacts.

## Architectural Constraints

- Custom strategy compatibility must remain intact.
- Diagnostics must not leak oracle labels.
- Diagnostics must be JSON-serializable primitives.
- Existing exact snapshot tests should only change when fallback diagnostics are expected.

## Acceptance Criteria

- Cold-start entropy and BADGE fallback snapshots expose effective strategy and reason.
- Normal single-strategy snapshots remain `{"mode": "single", "strategy": name}` in non-fallback paths.
- Benchmark/docs explicitly distinguish stochastic/committee proxy evidence from true stochastic/committee quality evidence.
- Focused tests and full suite pass.

## Suggested Validation

```powershell
uv run pytest tests\test_badge_strategy.py tests\test_class_balanced_entropy_strategy.py tests\test_group_diverse_strategy.py tests\test_core_refactor_characterization.py tests\test_benchmark_evidence_contract.py tests\test_sdk_first_benchmark_embedding_diagnostics.py -q
uv run pytest -q
uv run mypy src
uv run --with ruff ruff check .
```
