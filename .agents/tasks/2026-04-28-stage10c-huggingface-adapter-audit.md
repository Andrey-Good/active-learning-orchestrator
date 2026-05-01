# Stage 10C: Hugging Face Adapter Scope Audit

## Task Identifier

stage10c-huggingface-adapter-audit

## Context

The Hugging Face adapter is currently scaffold-like. Stage 10 should either make
its contract safer or document it honestly so users do not mistake it for a full
training adapter.

## Goal

Audit the Hugging Face adapter and docs for honest public-beta scope.

## Responsibility Boundaries

In scope:

- `HFSequenceClassifierAdapter` runtime behavior;
- optional dependency failure messages;
- `predict_proba` shape/probability validation;
- device handling and tokenizer/model edge cases;
- unsupported `fit`/`evaluate` capability inspection;
- docs/README statements about Hugging Face support.

Out of scope:

- Implementing full HF fine-tuning unless a narrow safe fix is clearly required
  later by the orchestrator.
- Changing sklearn adapter.
- Benchmark expansion.

## Files May Be Read

- `src/active_learning_sdk/adapters/huggingface.py`
- `src/active_learning_sdk/adapters/base.py`
- `tests/**`
- `README.md`
- `docs/**`
- `pyproject.toml`

## Files May Be Changed

- Only `.agents/tmp/2026-04-28-stage10c-huggingface-adapter-audit.md`

## Files Must Not Be Touched

- Production source files.
- Public docs outside the tmp report.
- Tests.

## Architectural Constraints

- Optional Hugging Face dependencies must not be required for core import.
- Scaffold methods must fail fast during capability inspection.
- Docs must not imply fit/evaluate support if the adapter does not provide it.

## Special Attention

- `predict_proba` should validate finite, non-negative, normalized rows.
- Tokenizer outputs and model tensors may need device movement in real use.
- `get_model_id` may be too weak for mutable trained models.

## Forbidden Actions

- Do not edit code.
- Do not add a fake training loop.
- Do not claim full adapter support without tests.

## Execution Plan

1. Inspect adapter behavior and public docs.
2. Identify correctness, safety, and claim-scope gaps.
3. Recommend either concrete hardening or explicit docs/tests boundaries.
4. Write severity-ranked report.

## Acceptance Criteria

- Report states whether HF adapter is acceptable as scaffold-only beta surface.
- Blockers are concrete and actionable.
- Non-blocking improvements are clearly separated.

## Expected Validations

- Read-only audit. Avoid installing heavy optional dependencies.

## Dependencies

- None.

## Parallelism

Can run in parallel with Stage 10A and Stage 10B.
