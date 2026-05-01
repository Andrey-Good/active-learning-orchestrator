# R91 - Review W72 Backend Contracts And HTTP Robustness

## Context

W72 added backend contract tests, Label Studio HTTP retry/backoff/strict JSON handling, finite-score filtering, and simulator oracle modes.

## Goal

Review W72 for correctness, product reliability, and absence of hidden test-only shortcuts before Stage 7 continues to managed Docker and timeout workflows.

## Responsibility Boundaries

You are a reviewer. Do not edit files.

## In Scope

- `src/active_learning_sdk/backends/label_studio.py`
- `src/active_learning_sdk/backends/simulator.py`
- `tests/test_label_backends.py`
- Any directly referenced backend contracts in `src/active_learning_sdk/backends/base.py`

## Out Of Scope

- Managed Docker runtime/assets.
- Engine timeout enforcement.
- README updates.

## Required Review Questions

- Are Label Studio retries bounded and limited to transient failures?
- Are permanent 4xx errors non-retryable with useful error messages?
- Are HTTP payloads strict-JSON-safe before network calls?
- Are non-finite prediction/annotation scores skipped or normalized safely?
- Is task idempotency based on stable round/sample metadata?
- Do simulator oracle modes preserve manual submit behavior and deterministic task ids?
- Do tests prove meaningful backend contracts without live services or sleeps?

## Explicitly Forbidden

- Do not edit files.
- Do not broaden the task.

## Validation To Run

- `uv run --group dev pytest -q tests/test_label_backends.py`
- `uv run --group dev pytest -q`

## Output

Return severity-ordered findings with file/line refs. If no findings remain, say so explicitly and include validation results.
