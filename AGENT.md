# AGENT

## Purpose

This file stores only stable working rules for the coding agent in this repository.
Keep it concise, practical, and free of temporary task notes.

## Working Rules

1. Before any large task with multiple dependent steps, create a detailed written plan.
2. Store every such plan only in the root `plans/` directory.
3. Do not scatter plan documents across the repository.
4. Keep `AGENT.md` clean: remove outdated, duplicated, or task-specific noise instead of appending endlessly.

## Plan Requirements

For every large multi-step task, create a plan document before implementation starts.

Each plan must include:
- goal and expected outcome;
- assumptions and constraints;
- execution order;
- stages/milestones;
- what must be done;
- what must not be done;
- important risks, checks, and edge cases;
- validation/acceptance criteria.

If the work changes materially, update the plan so it stays accurate.

## Code Quality

1. Write code at a senior-review level.
2. Prefer clear, maintainable solutions over quick hacks.
3. Do not leave dead code, temporary scaffolding, or avoidable technical debt.
4. Use workarounds only when there is no reasonable cleaner option, and keep them contained.
5. Keep implementations coherent with the existing architecture and repository style.

## Execution Discipline

1. Think through the order of changes before editing files.
2. Check important assumptions against the codebase instead of guessing.
3. Keep changes minimal but complete.
4. Preserve user changes and unrelated repository state.
5. Verify important behavior after changes whenever feasible.

## Repository Invariants

1. `ActiveLearningProject` is the public entrypoint; do not bypass it in user-facing solutions unless there is a strong reason.
2. `ActiveLearningEngine` is the orchestration core; keep facade, engine, strategies, backends, adapters, and dataset layers separated.
3. The round flow is a strict state machine:
   `SELECTING -> SELECTED -> PUSHED -> WAITING -> READY_TO_PULL -> PULLED -> TRAINED -> DONE`
4. Crash-safe resume depends on `RoundState.task_ids` being persisted after `PUSH`; do not break this idempotency invariant.
5. Strategies should work through `SelectionContext`; treat it as read-only and avoid coupling strategies directly to engine internals.
6. Model integrations should follow the adapter contract; for MVP the required methods are `predict_proba`, `fit`, and `evaluate`.
7. Configuration belongs in dataclasses from `configs.py`; avoid introducing ad hoc config formats when existing config objects fit.
8. Dataset integrity and resume safety matter: keep fingerprinting, state persistence, and locking behavior coherent.

## Implementation Status

Current stable areas:
- state machine and resume logic;
- uncertainty strategies (`random`, `entropy`, `least_confidence`, `margin`);
- annotation aggregation;
- dataset fingerprinting;
- caching;
- file locking;
- DataFrame dataset provider;
- synchronous `LLMLabelBackend`.

Current scaffolds / incomplete areas:
- `LabelStudioBackend`;
- `HFSequenceClassifierAdapter.fit`;
- `HFSequenceClassifierAdapter.evaluate`;
- report generation;
- `KCenterGreedyStrategy`.

When planning or implementing work, check whether the target area is real logic or scaffold code first.

## Subagents

Use subagents deliberately when they provide clear value:
- parallelizable independent analysis;
- disjoint implementation work;
- context savings on bounded side tasks.

Do not create subagents by default or for tightly coupled critical-path work.

## CLAUDE Sync Rules

`CLAUDE.md` may contain useful repo knowledge, but `AGENT.md` must only keep compact operational guidance.

## Source Priority And Freshness

When sources disagree, use this priority order:
1. direct user instructions in the current conversation;
2. newer validated rules already present in `AGENT.md`;
3. current codebase reality and repository files;
4. `CLAUDE.md` as a reference source.

`CLAUDE.md` is not authoritative over `AGENT.md`. It is an input for sync, not a reason to roll `AGENT.md` back to an older state.

If `AGENT.md` contains guidance that is newer, more specific, or clearly aligned with the current repository/user workflow, keep it unless there is strong evidence it is outdated.

Treat information as newer/more authoritative when at least one of these is true:
- it was added to `AGENT.md` from direct user instruction;
- it reflects current repository structure or implementation better than `CLAUDE.md`;
- it resolves a gap, correction, or stale statement from `CLAUDE.md`;
- it matches recent changes in code, workflow, or project layout.

When checking `CLAUDE.md` for updates, copy information into `AGENT.md` only if it is all of the following:
- stable across tasks;
- directly useful for implementation decisions;
- specific to this repository;
- concise enough to keep `AGENT.md` clean.

Do copy:
- public API and layering invariants;
- state-machine and idempotency invariants;
- required adapter/backend/strategy constraints;
- implementation-status changes that affect what is safe to build on;
- repository-specific workflow rules that change how work should be done.

Do not copy:
- long architecture walkthroughs;
- roadmap or PRD target ideas;
- speculative future features;
- broad project summaries;
- examples, tutorials, or documentation inventory;
- anything temporary, redundant, or better kept in guides/README.

For the daily sync agent:
1. Compare `CLAUDE.md` against the current `AGENT.md`.
2. Extract only new or changed high-signal operational facts.
3. Skip descriptive or low-signal text.
4. Check source priority before changing anything; do not let `CLAUDE.md` override newer `AGENT.md` guidance.
5. Merge information by updating existing bullets when possible instead of appending duplicates.
6. Remove or replace outdated entries only when codebase reality or direct user guidance confirms they are outdated.
7. If `CLAUDE.md` conflicts with `AGENT.md` and freshness is unclear, prefer keeping `AGENT.md` unchanged.
8. Never downgrade a rule in `AGENT.md` just because `CLAUDE.md` has an older or broader version.
9. Never modify `CLAUDE.md` during this sync.
10. If nothing important changed, leave `AGENT.md` untouched.
