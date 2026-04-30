# W95 Repeat Acceptance Blockers

## Context
Repeat senior acceptance left four blockers: invalid embeddings can be cached, custom selectors cannot inspect the candidate pool, hybrid fallback count is underreported, and README/audit artifacts have stale evidence.

## Goal
Fix the remaining blockers without weakening the W94 runtime/state fixes.

## Ownership Split
- Cache/selector changes: `src/active_learning_sdk/engine.py`, related tests.
- Hybrid fallback reporting: `src/active_learning_sdk/strategies/hybrid.py`, related tests.
- Evidence refresh: `README.md`, relevant audit docs/results, benchmark artifacts.

## Acceptance Criteria
- `tests/test_repeat_acceptance_blockers.py` has no xfail and passes normally.
- A custom selector can inspect the current unique candidate pool.
- Hybrid snapshots report actual fallback usage.
- Invalid embeddings are rejected before `EmbeddingCache.set`.
- README and current audit evidence report current test counts and current benchmark evidence.
- Full suite remains green.

## Constraints
- Do not revert unrelated work.
- Preserve backward compatibility for existing two-argument custom selectors if practical.
- Do not silently overwrite benchmark artifacts unless using explicit `--overwrite`.
- Keep historical rejected reports clearly marked as historical if not fully rewritten.
