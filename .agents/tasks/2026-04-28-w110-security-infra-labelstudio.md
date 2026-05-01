# Task W110 - Security/Infra/Label Studio Fixes

## Context

The 2026-04-28 objection sweep added failing tests for Label Studio HTTP retry semantics, annotation readiness, managed Docker credentials/bindings, and Docker Compose timeout handling.

## Goal

Fix security and infrastructure failures without touching unrelated SDK areas.

## Ownership

May edit:
- `src/active_learning_sdk/backends/label_studio.py`
- `src/active_learning_sdk/backends/managed_docker.py`
- `src/active_learning_sdk/backends/assets/label_studio/docker-compose.yml`
- related tests only if implementation reveals clearly wrong expectations, but prefer production fixes.

Must not edit:
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/configs.py`
- `src/active_learning_sdk/cache.py`
- `src/active_learning_sdk/state/*`
- `src/active_learning_sdk/strategies/*`

## Failing Tests To Fix

- `tests/test_objection_sweep_security_infra_2026_04_28.py::test_http_client_does_not_retry_non_idempotent_post_after_transient_500`
- `tests/test_objection_sweep_security_infra_2026_04_28.py::test_poll_round_does_not_count_unparseable_label_studio_annotations_as_ready`
- `tests/test_objection_sweep_security_infra_2026_04_28.py::test_managed_runtime_requires_explicit_token_or_secret_env`
- `tests/test_objection_sweep_security_infra_2026_04_28.py::test_packaged_managed_label_studio_proxy_binds_loopback_only`
- `tests/test_objection_sweep_security_infra_2026_04_28.py::test_compose_command_wraps_docker_version_timeout_as_infrastructure_error`

## Requirements

- Retry only idempotent HTTP methods by default; do not replay POST after transient errors unless a real idempotency mechanism is present.
- Poll readiness must count parsed, schema-valid annotations, not raw annotation shells.
- Managed Docker must not fall back to static credentials/token. Require explicit env/config secrets or generate/persist runtime-local secrets if implementing that path safely.
- Packaged compose must bind Label Studio proxy to loopback by default.
- Wrap Docker Compose probe/startup timeouts as `InfrastructureError`.

## Validation

Run:

```powershell
uv run pytest tests\test_objection_sweep_security_infra_2026_04_28.py -q
```

Also run a syntax/static sanity command for touched modules.

## Notes

You are not alone in the codebase. Do not revert or rewrite unrelated edits made by other agents.
