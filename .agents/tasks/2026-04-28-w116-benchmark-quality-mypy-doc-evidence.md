# Task W116: quality-gate parser, sklearn mypy policy, evidence/docs hygiene

## Context

The current open-defect sweep includes:

- `quality_gate_report.py` crashes on string metadata columns such as `strategy_family`;
- `uv run mypy src` fails on untyped sklearn imports;
- docs/backlog validation counters and evidence claims need to match current remediation after tests are green.

## Goal

Fix the executable benchmark parser defect, make the obvious mypy command green, and update current evidence docs only after validation counts are known.

## Ownership

You may edit:

- `benchmarks/quality_gate_report.py`
- `src/active_learning_sdk/adapters/sklearn.py`
- `pyproject.toml`
- `README.md`
- `docs/SENIOR_SDK_ALL_CURRENT_OPEN_OBJECTIONS_2026-04-28.md`
- other release-facing docs only for validation-count/evidence updates

Do not edit runtime state/cache/fingerprint/split logic or Label Studio backend.

## Constraints

- Preserve numeric parsing for real metric columns; string metadata must be retained or ignored without crashing.
- Prefer project mypy config for optional untyped dependencies over noisy inline ignores unless inline ignores are clearer.
- Do not claim current full-suite counts until the orchestrator has run them.

## Forbidden Actions

- Do not delete benchmark artifacts to hide evidence gaps.
- Do not relax mypy so broadly that SDK source stops being checked.
- Do not remove current-open backlog items unless the orchestrator confirms the relevant fix landed.

## Suggested Validation

- `uv run pytest tests\test_current_open_audit_defects_2026_04_28.py -q --runxfail`
- `uv run mypy src`
- `uv run --with ruff ruff check .`

## Acceptance Criteria

- The quality-gate parser xfail passes under `--runxfail`.
- `uv run mypy src` is green or the supported type-check policy is explicit and CI-aligned.
- Release-facing docs are consistent with final validation.
