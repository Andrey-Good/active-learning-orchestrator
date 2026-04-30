# Task: blackbox-stress-wave2-load-concurrency

## Context
Second wave of the user's aggressive black-box SDK stress test. Source-code inspection is forbidden; use documentation and public runtime behavior only.

## Goal
Stress the SDK under larger pools, repeated runs/resumes, concurrent access, slow/failing backends/models, cache pressure, report generation, and stop/timeout policies.

## Responsibility Boundaries
Owns only artifacts under `.agents/tmp/blackbox_stress/wave2_load/` and findings in `.agents/tmp/blackbox_stress/wave2_load_findings.md`.

## In Scope
- Read only public documentation.
- Write external harnesses under owned temp dir.
- Use public `ActiveLearningProject`, public configs, simulator/custom backends, and user-side models/providers.
- Attack: thousands of samples, long sample IDs, duplicate-ish group IDs, repeated resume cycles, process lock conflicts, timeout handling, backend partial labels, model latency, report/export under larger state, cache stats/clear behavior, and deterministic reproducibility.

## Out of Scope
- Reading SDK source.
- Modifying SDK source/tests/docs/benchmarks.
- Fixing bugs.

## Must Not Touch
- `src/**`
- `tests/**`
- `benchmarks/**`
- other stress worker directories

## Execution Plan
1. Create a black-box load/concurrency harness.
2. Run bounded but adversarial cases that finish locally.
3. Prefer hard evidence: elapsed time, row counts, state counts, exceptions, artifact sizes.
4. Classify issues by severity.

## Acceptance Criteria
- At least 15 load/concurrency/state cases attempted.
- At least one case uses >= 2,000 samples.
- At least one case exercises timeout/partial-label behavior.
- At least one case exercises independent process lock or concurrent open behavior.
