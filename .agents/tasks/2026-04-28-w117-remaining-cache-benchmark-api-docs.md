# Task W117: remaining current-open objections after executable fixes

## Context

W114-W116 fixed the 8 executable xfail defects from `tests/test_current_open_audit_defects_2026_04_28.py`. The current-open backlog still contains non-xfail objections around cache scalability/contract, benchmark evidence wording, benchmark probability strictness, public API narrowness, pycache hygiene, and audit-doc discoverability.

## Goal

Close the remaining low/medium-risk objections that can be safely fixed without broad architecture refactors or native external benchmark research.

## Ownership

You may edit:

- `src/active_learning_sdk/cache.py`
- `src/active_learning_sdk/__init__.py`
- `benchmarks/audit_sdk_vs_manual.py`
- `benchmarks/README.md`
- `README.md`
- `docs/SENIOR_SDK_ALL_CURRENT_OPEN_OBJECTIONS_2026-04-28.md`
- add a small docs index if helpful
- add focused tests if necessary

Do not edit engine/state/fingerprint/Label Studio fixes from W114/W115.

## Constraints

- JSONL cache should avoid O(N) line counting on every `set()` and state its single-writer/crash-recovery contract.
- Benchmark invalid probability rows should be rejected consistently with SDK strict probability contracts.
- Docs must not overclaim that formula-shim benchmarks are native modAL/scikit-activeml workflow evidence.
- Public API exports should be deliberate and tested if changed.
- Do not claim architectural complexity refactors or native external workflow benchmarks are complete unless actually implemented.

## Forbidden Actions

- Do not delete real benchmark evidence to hide limitations.
- Do not relax tests or validation.
- Do not perform destructive git commands.

## Suggested Validation

- `uv run pytest -q`
- `uv run mypy src`
- `uv run --with ruff ruff check .`
- any focused benchmark parser/probability tests you add

## Acceptance Criteria

- Safe objections are fixed or explicitly reclassified as remaining roadmap items.
- Current evidence in README/current backlog matches actual validation.
- No new failures in full test/static/build gates.
