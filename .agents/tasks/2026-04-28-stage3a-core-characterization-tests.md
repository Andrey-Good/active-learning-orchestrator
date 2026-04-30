# Stage 3A: Core Refactor Characterization Tests

## Context

Stage 3 is core architecture refactoring. Before moving code, we need focused characterization tests that pin current behavior for high-risk engine areas.

Hotspots from radon:

- `ActiveLearningEngine._validate_persisted_splits`
- `ActiveLearningEngine._resolve_splits`
- `SelectionContext.predict_proba`
- `SelectionContext.embed`
- `StrategyScheduler.select_batch`
- `ActiveLearningEngine.validate`

## Goal

Add tests that characterize key behavior without changing production code. These tests should make later extraction/refactor safe.

## Ownership

You may edit/add tests only:

- `tests/test_core_refactor_characterization.py`
- or small additions to existing focused tests if strongly preferable

Do not edit production SDK code.

## In Scope

Add tests covering:

- explicit split coverage/overlap/unknown-id behavior;
- column split stability behavior if easy;
- prediction cache scoping and invalid cached row eviction;
- embedding cache scoping and invalid cached row eviction;
- scheduler duplicate/out-of-pool protection on custom strategy return values;
- `validate()` returning a structured report for a healthy configured project.

## Out Of Scope

- Production refactor.
- New strategy behavior.
- Benchmark changes.

## Constraints

- Keep tests fast and deterministic.
- Prefer local fake providers/models/backends.
- Tests should not require optional external dependencies.

## Suggested Validation

- `uv run pytest tests/test_core_refactor_characterization.py -q`
- `uv run pytest -q`

## Acceptance Criteria

- New characterization tests pass.
- They cover the highest-risk current behavior before extraction begins.
