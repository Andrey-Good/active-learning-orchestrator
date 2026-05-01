# 2026-04-28-w07-adapters-config-api-objection-sweep

## Context

Second-pass exhaustive senior audit requested on 2026-04-28.

## Goal

Audit model adapters, configuration validation, public API consistency, optional dependency boundaries, data-provider contracts, and README/API mismatch not already covered by W01.

## Responsibility Boundaries

Owner may change only:

- `tests/test_objection_sweep_adapters_config_api_2026_04_28.py`
- `.agents/tmp/2026-04-28-w07-adapters-config-api-findings.md`

Owner must not change:

- `src/**`
- existing tests
- benchmark files
- docs except the owned findings file

## In Scope

- `adapters/base.py`
- `adapters/sklearn.py`
- `adapters/huggingface.py`
- `configs.py`
- `dataset/provider.py`
- public exports and facade behavior
- README snippets when directly tied to actual API

## Out of Scope

- Deep scheduler math.
- Runtime backend orchestration.
- Benchmark methodology.

## Constraints

- Do not require torch/transformers downloads.
- Mock optional dependencies where needed.
- Keep tests fast.

## Acceptance Criteria

- Findings are reproducible and actionable.
- Separate product limitations from defects.
- Include severity and fix direction.

## Dependencies

Can run in parallel with W05, W06, W08, and W09.
