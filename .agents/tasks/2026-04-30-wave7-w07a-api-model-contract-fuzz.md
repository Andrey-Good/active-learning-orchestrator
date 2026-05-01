# W07A - API And Model Contract Fuzz

## Task Identifier

W07A-API-MODEL-CONTRACT-FUZZ

## Context

Wave7 is a documentation-only black-box SDK stress campaign. This subtask attacks public API, configuration, model adapter capability validation, probability payload validation, scheduler configuration, state copy-safety, and runtime exception taxonomy.

## Goal

Find reproducible public API/model-contract defects without inspecting SDK source.

## Responsibility Boundaries

May change/write only:
- `.agents/tmp/blackbox_stress_wave7/w07a_api_model/**`

Must not touch:
- `src/active_learning_sdk/**`
- `tests/**`
- `benchmarks/*.py`
- existing docs, package metadata, lockfiles, or existing generated artifacts outside this task directory.

## In Scope

- Public root exports from `docs/SDK_CONTRACTS.md`.
- `ActiveLearningProject` black-box flows using custom providers/models/backends.
- Adversarial `predict_proba`, `predict_stochastic`, `predict_committee`, `embed`, `gradient_embed`, `fit`, `evaluate`, and `get_model_id` behavior.
- Scheduler modes and strategy configuration through documented names.
- Exception taxonomy and returned object copy-safety.

## Out Of Scope

- Quality benchmarking across real datasets.
- Packaging/venv install tests.
- Live Label Studio service tests.

## Architectural Constraints

Use public imports and documented methods only. If a behavior is undocumented, treat it as exploratory and do not overstate it as a defect unless it violates a stable contract.

## Special Attention

Reject false positives caused by invalid user models unless the SDK promised validation/taxonomy and leaks raw errors, corrupts state, hangs, or silently accepts invalid payloads.

## Execution Plan

1. Read only README/docs/package metadata needed for public contract.
2. Create a standalone fuzz harness under the task directory.
3. Run targeted cases for malformed probabilities, weird sample ids/texts, scheduler configs, model capability traps, and resume/state copy-safety.
4. Write `results.json`, optional CSV/logs, and `findings.md`.

## Acceptance Criteria

- At least 50 meaningful cases attempted.
- Findings include command, exact case id, observed exception/result, expected behavior from docs, severity, and repro notes.
- No SDK source or repository tests are inspected.

## Dependencies

Can run in parallel with W07B/W07C/W07D.
