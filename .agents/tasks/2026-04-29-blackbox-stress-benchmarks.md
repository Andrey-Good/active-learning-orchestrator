# Task: blackbox-stress-benchmarks

## Context
The user requested broad SDK stress across datasets, networks/models, and SDK settings. Source-code inspection is forbidden. Documentation and black-box execution only.

## Goal
Use documented benchmark entrypoints and external scripts to expose low metrics, brittle strategy behavior, runtime/performance cliffs, and benchmark-contract gaps.

## Responsibility Boundaries
Owns only artifacts under `.agents/tmp/blackbox_stress/benchmarks/` and notes in `.agents/tmp/blackbox_stress/benchmark_findings.md`.

## In Scope
- Read only documentation: `README.md`, `docs/BENCHMARK_EVIDENCE.md`, `benchmarks/README.md`, and other docs when needed.
- Run documented benchmark commands with small but varied matrices.
- Test multiple datasets, seeds, budgets, and strategies, including real datasets if dependencies/network allow.
- Compare metrics versus random baseline, calibration fields, runtime, duplicate/group concentration, and manifest evidence fields.
- Create external wrapper scripts under owned temp dir if needed.

## Out of Scope
- Reading SDK implementation files under `src/**`.
- Modifying SDK source or benchmark source.
- Presenting capped or smoke results as universal scientific proof.

## Must Not Touch
- `src/**`
- existing benchmark source files
- existing promoted benchmark results

## Architectural Constraints
Use documented CLI surfaces first. Keep run sizes bounded so the main orchestration can finish, but choose adversarial settings: tiny budgets, imbalanced data, diverse strategy families, multiple seeds, and at least one real-dataset attempt.

## Execution Plan
1. Run a documented synthetic smoke or custom benchmark into owned output dir.
2. Run at least one adversarial custom synthetic matrix with low budgets and strategy mix.
3. Attempt a real-data smoke or standard-real miniature only if dependencies/network cooperate.
4. Parse artifacts and summarize weak metrics, failures, skips, and contract problems.

## Acceptance Criteria
- At least 2 benchmark commands attempted.
- At least 3 strategy families tested.
- Findings include paths to CSV/JSON/Markdown artifacts and clear metric thresholds or comparisons.

## Dependencies
Can run in parallel with API/state and model-adapter stress tasks. No shared write scope.
