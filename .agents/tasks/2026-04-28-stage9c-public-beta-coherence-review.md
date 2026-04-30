# Review Stage 9C: Public Beta Coherence Fixes

## Context

Stage 9C fixed public beta docs/package coherence findings from Stage 9A/9B audits.

## Goal

Review whether the public beta coherence blockers are actually closed without introducing overclaims or packaging regressions.

## Read Scope

- `.agents/tmp/2026-04-28-stage9a-public-beta-docs-audit.md`
- `.agents/tmp/2026-04-28-stage9b-public-beta-packaging-examples-audit.md`
- `README.md`
- `docs/README.md`
- `docs/SDK_REAL_PRODUCT_ROADMAP.md`
- `benchmarks/README.md`
- `docker/label_studio/README.md`
- `docker/label_studio/docker-compose.yml`
- `src/active_learning_sdk/backends/assets/label_studio/docker-compose.yml`
- `pyproject.toml`
- `tests/test_managed_label_studio.py`
- `tests/test_public_beta_docs_package_coherence.py`

## Write Scope

- `.agents/tmp/2026-04-28-stage9c-public-beta-coherence-review.md` only

## Review Questions

- Is the simulator quickstart smoke-testable from the documented install path?
- Are extras and feature mappings complete and consistent with `pyproject.toml`?
- Do managed Docker docs, repo compose, packaged compose, and tests agree on explicit credentials?
- Are benchmark claim categories consistent?
- Is roadmap/status framing coherent and not overclaiming?
- Is package metadata appropriate for public beta?
- Any new P1/P2 findings?

## Constraints

- Review only, do not edit.

## Suggested Validation

```powershell
uv run pytest tests\test_public_contracts.py tests\test_optional_dependency_packaging.py tests\test_project_smoke_benchmark.py tests\test_sklearn_adapter.py tests\test_managed_label_studio.py tests\test_public_beta_docs_package_coherence.py -q
uv run pytest -q
uv run mypy src
uv run --with ruff ruff check .
uv build
uv run twine check dist\active_learning_sdk-0.1.0.tar.gz dist\active_learning_sdk-0.1.0-py3-none-any.whl
```

## Acceptance Criteria

- Explicit accept/reject.
- If accepted, Stage 9 public beta coherence can close.
