# W08A Public API And Taxonomy Stress

Task identifier: `wave8-w08a-public-api-taxonomy`

## Goal

Stress the public SDK facade, model adapter contract, custom backend contract, scheduler/prelabel probability validation, and public exception taxonomy using only black-box runtime behavior.

## Ownership

May write only under `.agents/tmp/blackbox_stress_wave8/w08a_api_taxonomy/`.

Must not touch product files, docs, tests, benchmark source, or other workers' directories.

## In Scope

- Public imports from `active_learning_sdk`.
- Runtime scripts that instantiate `ActiveLearningProject` with tiny custom providers, models, and backend objects.
- Malformed model outputs for `predict_proba`, stochastic, committee, embeddings, and model ids where public docs define behavior.
- Custom backend lifecycle failures and malformed backend return payloads.
- Public exception category checks against `docs/SDK_CONTRACTS.md`.
- Recheck historical Wave7 taxonomy findings against the current worktree.

## Out Of Scope

- Reading implementation source under `src/**`.
- Reading repository tests under `tests/**`.
- Live Label Studio.
- Editing SDK code.

## Plan

1. Create a self-contained harness script under the owned tmp directory.
2. Run valid quickstart-like baseline first to prove the setup works.
3. Run adversarial cases with one isolated workdir per case.
4. Record observed exception type, message, and whether it is an `ActiveLearningError` subtype.
5. Re-run any candidate finding at least once independently or through a minimal reproduction script.
6. Write `findings.md` and machine-readable `results.json`.

## Acceptance Criteria

- Findings include only reproducible current-worktree behavior.
- Each issue includes reproduction command, expected behavior, actual behavior, severity, confidence, and fix/regression direction.
- Cases that behave correctly or are rejected are summarized briefly.
