# 2026-04-29 Wave6 W06B Cache Resume Load Stress

## Task Identifier

W06B-CACHE-RESUME-LOAD-STRESS.

## Context

Earlier black-box waves found and then reportedly fixed Windows persistent-cache failures and several state/cache issues. This worker tries to break cache/resume/load behavior again with larger and adversarial public workflows.

## Goal

Find reproducible cache, filesystem, resume, lock, and lifecycle defects using public APIs only.

## Ownership

May write only under `.agents/tmp/blackbox_stress_wave6/w06b_cache_resume/`.

## In Scope

- Persistent and non-persistent prediction/embedding cache behavior.
- Larger entropy/diversity runs with model IDs that change, stay stable, or are absent.
- Reopen/attach cycles, repeated report/export/cache-clear cycles.
- Concurrent process or lock probes through public project open/configure/run behavior.
- Crash-like partial artifact probes by manipulating only generated workdirs, not SDK code.
- Performance timings and cache stats sanity checks.

## Out Of Scope

- Reading `src/active_learning_sdk/**`.
- Reading repository `tests/**`.
- Modifying SDK, docs, benchmarks, or lockfiles.
- Destructive cleanup outside the owned artifact directory.

## Important Constraints

- Do not use private SDK state attributes.
- Generated workdirs must remain under the owned directory.
- Any process-level manipulation must be bounded with timeouts.

## Execution Plan

1. Build a standalone public-API stress script.
2. Run cache/load/resume scenarios at several dataset sizes.
3. Capture timings, cache stats, exception taxonomy, and generated artifacts.
4. Write findings with severity and replay commands.

## Acceptance Criteria

- At least 15 cache/resume/load scenarios run.
- Persistent-cache stats are compared against observed predictions or documented expectations.
- Any concurrency issue includes process command and timeout details.
