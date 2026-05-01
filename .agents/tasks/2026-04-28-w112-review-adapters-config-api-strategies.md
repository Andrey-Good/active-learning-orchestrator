# Task W112 Review - Adapters/Config/API/Strategies

## Goal

Read-only senior review of W112 fixes plus the integration repair for capability diagnostics.

## Scope

Review:
- `pyproject.toml`
- `.github/workflows/ci.yml`
- `src/active_learning_sdk/adapters/__init__.py`
- `src/active_learning_sdk/adapters/base.py`
- `src/active_learning_sdk/adapters/huggingface.py`
- `src/active_learning_sdk/configs.py`
- `src/active_learning_sdk/engine.py` only prelabel/capability-related changes
- `src/active_learning_sdk/strategies/uncertainty.py`
- `src/active_learning_sdk/strategies/hybrid.py`
- related 2026-04-28 adapter/config/strategy tests.

## Checkpoints

- `xxhash` extra exists and is included in `all`.
- Missing sklearn public adapter error is actionable.
- Minimal runtime protocol remains compatible while capability diagnostics still reject protocol stubs.
- Config validators reject invalid types/ranges with `ConfigurationError`.
- HF zero batch size behavior is correct.
- Prelabels validate probability rows against label schema before backend push.
- Group metadata lookups fail closed on misordered/missing/foreign sample ids.
- CI/dev tooling is encoded without hardcoding brittle local paths.

## Do Not

- Edit files.
- Review unrelated Label Studio/cache/lock changes.

## Validation Context

The orchestrator observed:
- W112 focused tests -> `15 passed`
- all 2026-04-28 audit tests -> `27 passed`
- full suite -> `431 passed`
- Ruff/Mypy/build/twine -> pass

Report blockers with file/line references or state no blockers remain for this scope.
