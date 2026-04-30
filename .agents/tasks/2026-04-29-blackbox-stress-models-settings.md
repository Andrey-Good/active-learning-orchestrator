# Task: blackbox-stress-models-settings

## Context
The user asked to test the SDK with different networks/models and SDK settings while treating the SDK as a black box. Source-code inspection is forbidden.

## Goal
Stress model adapter capability validation and strategy compatibility using multiple external model shapes: deterministic models, malformed models, sklearn models, embedding/logit/stochastic/committee-capable adapters, and intentionally slow/failing adapters.

## Responsibility Boundaries
Owns only artifacts under `.agents/tmp/blackbox_stress/models_settings/` and notes in `.agents/tmp/blackbox_stress/model_settings_findings.md`.

## In Scope
- Use public docs to build consumer-side adapters.
- Exercise `inspect_model_capabilities`, scheduler configurations, prelabel settings, embedding strategies, BADGE/stochastic/committee strategies, custom schedulers, cache configs, fingerprint configs, and split configs.
- Use sklearn or downloaded models/datasets if already available or installable through existing environment.
- Prefer small bounded datasets for speed.

## Out of Scope
- Reading SDK source.
- Modifying SDK source, tests, or docs.
- Depending on private SDK internals.

## Must Not Touch
- `src/**`
- `tests/**`
- existing benchmark source/results

## Architectural Constraints
Each issue must be framed as public-contract or documentation mismatch, runtime robustness failure, or low-quality metric behavior.

## Execution Plan
1. Create external adapter stress script under owned temp dir.
2. Exercise valid and invalid adapter capability combinations across strategy settings.
3. Run at least one sklearn-backed project loop if dependencies allow.
4. Summarize failures, weak validations, misleading errors, and model-setting incompatibilities.

## Acceptance Criteria
- At least 15 model/settings combinations attempted.
- At least one probability strategy, one embedding/diversity strategy, one stochastic/committee strategy, and one custom scheduler path attempted.
- Findings include reproduction command and expected/observed behavior.

## Dependencies
Can run in parallel with other black-box stress tasks. No shared write scope.
