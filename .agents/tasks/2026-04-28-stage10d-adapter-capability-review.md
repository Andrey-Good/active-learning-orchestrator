# Stage 10D Review: Adapter And Capability Fixes

## Task Identifier

stage10d-adapter-capability-review

## Context

Stage 10D worker implemented fixes for audit blockers from Stage 10A/10B/10C.
This is an independent review. Do not assume the implementation is correct
because tests passed.

## Goal

Review the Stage 10D changes for correctness, maintainability, API safety, and
test adequacy.

## Responsibility Boundaries

In scope:

- changes in engine/project custom strategy registration and validation;
- sklearn adapter model identity, one-class rejection, optional import handling;
- Hugging Face adapter probability validation and device movement;
- adapter contract docs alignment;
- focused tests added/changed by Stage 10D.

Out of scope:

- Implementing fixes unless explicitly requested later.
- Reworking unrelated SDK subsystems.
- Benchmark expansion.

## Files May Be Read

- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/project.py`
- `src/active_learning_sdk/adapters/base.py`
- `src/active_learning_sdk/adapters/sklearn.py`
- `src/active_learning_sdk/adapters/huggingface.py`
- `tests/test_strategy_capabilities.py`
- `tests/test_sklearn_adapter.py`
- `tests/test_huggingface_adapter.py`
- `README.md`
- `docs/SDK_CONTRACTS.md`
- Stage 10A/10B/10C audit reports in `.agents/tmp`

## Files May Be Changed

- Only `.agents/tmp/2026-04-28-stage10d-adapter-capability-review.md`

## Files Must Not Be Touched

- Production source files.
- Tests.
- Public docs outside the review report.

## Review Questions

1. Are the original P1/P2 audit blockers actually closed?
2. Did the custom strategy API introduce inconsistent persistence/resume behavior?
3. Is sklearn fitted-state fingerprint deterministic and broad enough for common
   fitted estimators without becoming fragile or huge?
4. Does sklearn one-class rejection happen at all relevant paths?
5. Does direct optional import handling preserve useful stack traces while giving
   actionable install guidance?
6. Does HF device movement work with mapping-like tokenizer outputs and avoid
   moving non-tensor metadata incorrectly?
7. Does HF validation reject NaN/inf/negative/non-normalized/one-column outputs?
8. Are docs now honest about minimal adapter requirements vs strategy capabilities?

## Acceptance Criteria

- Write a verdict: accept or reject.
- List P1/P2 blockers first if any exist.
- Include exact file/line evidence for findings.
- If accepted, note any residual P3 risks.

## Expected Validations

Run focused tests if needed:

- `uv run pytest tests/test_strategy_capabilities.py tests/test_sklearn_adapter.py tests/test_huggingface_adapter.py -q`

Full suite/static checks are optional for reviewer if worker already ran them,
but mention if not rerun.

## Dependencies

- Stage 10D worker patch.

## Parallelism

Single reviewer task.
