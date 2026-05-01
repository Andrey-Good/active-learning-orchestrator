# Repeat Review Stage 4C: Hybrid KMeans++ Batch Fix

## Context

The first Stage 4B/4C review failed on one P2: hybrid weighted `embedding_kmeans_pp` lost greedy batch semantics for `k > 1`. A fix worker changed only `hybrid.py` and `tests/test_hybrid_strategy_framework.py`.

## Goal

Review whether the exact P2 is closed and whether the fix introduced new P1/P2 regressions.

## Read Scope

- `.agents/tmp/2026-04-28-stage4bc-strategy-quality-review.md`
- `src/active_learning_sdk/strategies/hybrid.py`
- `tests/test_hybrid_strategy_framework.py`
- related embedding strategy tests if needed

## Write Scope

- `.agents/tmp/2026-04-28-stage4c-repeat-review.md` only

## Review Questions

- Does weighted hybrid with pure `embedding_kmeans_pp` now use batch-level greedy diversity for `k > 1`?
- Does the test fail on the previous static centroid-representative ranking?
- Are weighted hybrid combinations with nonzero uncertainty still sane and deterministic?
- Any new public API, validation, or snapshot regressions?

## Constraints

- Review only, do not edit.

## Suggested Validation

```powershell
uv run pytest tests\test_hybrid_strategy_framework.py tests\test_embedding_strategies.py -q
uv run pytest -q
```

## Acceptance Criteria

- Explicit accept/reject.
- If accepted, Stage 4B/4C can be treated as complete.
