# Stage 9C: Public Beta Coherence Fixes

## Context

Stage 9A/9B audits rejected public beta coherence. The failures are documentation/package metadata/examples/asset parity issues, not strategy/runtime blockers.

## Goal

Fix the P1/P2 public beta coherence findings so install guidance, quickstarts, extras, managed Docker docs/assets, benchmark claim categories, roadmap/status framing, and package metadata align.

## Ownership

May edit:

- `README.md`
- `docs/README.md`
- `docs/SDK_REAL_PRODUCT_ROADMAP.md`
- `benchmarks/README.md`
- `docker/label_studio/docker-compose.yml`
- `docker/label_studio/README.md`
- `pyproject.toml`
- `tests/test_managed_label_studio.py`
- add a focused docs/examples test file if needed

Must not edit:

- strategy implementation files
- engine/runtime/state/cache/backend Python code except tests
- benchmark implementation scripts unless a docs-only test needs constants

## Required Fixes

- Replace or supplement the README "Minimal Usage Model" with a smoke-testable simulator quickstart that matches a documented install path.
- Keep Label Studio external/managed Docker flows separate from the first-run simulator path.
- Document all extras: `sklearn`, `huggingface`, `datasets`, `xxhash`, `benchmarks`, and `all`, with feature mapping.
- Align managed Docker README/sample/default credential wording with runtime behavior: explicit username/password/token secrets are required for packaged managed mode.
- Make repo `docker/label_studio/docker-compose.yml` match packaged compose credential contract, and extend asset parity tests to cover `docker-compose.yml`.
- Use `native_external_library_workflow_smoke` consistently as the native external claim category in docs.
- Update roadmap/status wording so current promoted smoke, retained Stage 9 diagnostics, and capped real-data diagnostics are clearly separated.
- Avoid fragile "latest dirty worktree" hard-coded claims in public-facing docs unless labeled as dated local validation.
- Update package classifier from Alpha to Beta if this is controlled public beta packaging.

## Acceptance Criteria

- Public docs contain a simulator quickstart that can be smoke-checked without pandas or live Label Studio.
- Managed Docker docs/assets/tests agree on explicit credential requirements.
- Extras docs match `pyproject.toml`.
- Benchmark claim category docs use consistent strings.
- Focused docs/package tests pass.
- Full suite/static/build remain green.

## Suggested Validation

```powershell
uv run pytest tests\test_public_contracts.py tests\test_optional_dependency_packaging.py tests\test_project_smoke_benchmark.py tests\test_sklearn_adapter.py tests\test_managed_label_studio.py -q
uv run pytest -q
uv run mypy src
uv run --with ruff ruff check .
uv build
uv run twine check dist\active_learning_sdk-0.1.0.tar.gz dist\active_learning_sdk-0.1.0-py3-none-any.whl
```
