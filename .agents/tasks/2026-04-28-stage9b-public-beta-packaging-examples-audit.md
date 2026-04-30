# Stage 9B: Public Beta Packaging/Examples Audit

## Context

Stage 9 public beta coherence also needs package/example validation, not just prose. Current suite includes public contract tests, optional dependency tests, and a project smoke benchmark using the public sklearn adapter.

## Goal

Read-only audit package metadata, optional extras, import behavior, build contents, and existing example/smoke tests. Identify concrete gaps that block a controlled public beta release.

## Read Scope

- `pyproject.toml`
- `src/active_learning_sdk/__init__.py`
- `src/active_learning_sdk/adapters/__init__.py`
- `src/active_learning_sdk/adapters/sklearn.py`
- `README.md`
- tests covering public contracts, optional dependencies, project smoke, sklearn adapter, managed docker
- built package metadata if useful through local commands

## Write Scope

- `.agents/tmp/2026-04-28-stage9b-public-beta-packaging-examples-audit.md` only

## In Scope

- Whether optional extras are documented and declared consistently.
- Whether root/adapters imports avoid eager optional dependencies.
- Whether built distributions include required assets and exclude obvious junk.
- Whether quickstart/project smoke coverage proves at least one public path.
- Whether Docker/Label Studio docs line up with package assets.

## Out Of Scope

- Do not edit code/tests/docs.
- Do not run long benchmarks.

## Suggested Validation

```powershell
uv build
uv run pytest tests\test_public_contracts.py tests\test_optional_dependency_packaging.py tests\test_project_smoke_benchmark.py tests\test_sklearn_adapter.py tests\test_managed_label_studio.py -q
```

## Acceptance Criteria

- Audit file exists with accept/reject and prioritized findings.
- Findings are reproducible and scoped for a follow-up worker.
