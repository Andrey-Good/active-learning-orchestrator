# Task Repeat Review - 2026-04-28 P1 Reviewer Findings

## Goal

Read-only repeat review of the P1 findings raised by the first reviewer pass.

## Findings Claimed Fixed

1. Label Studio readiness counted schema-invalid annotations.
2. Managed Docker fallback compose could bind publicly.
3. `ProjectLock.release()`/stale cleanup could delete a lock not owned by the instance.
4. Config validators could leak raw `TypeError`.
5. Prelabels accepted non-unit probability rows.

## Scope

Review:
- `src/active_learning_sdk/backends/label_studio.py`
- `src/active_learning_sdk/backends/managed_docker.py`
- `src/active_learning_sdk/state/lock.py`
- `src/active_learning_sdk/configs.py`
- `src/active_learning_sdk/engine.py`
- `docker/label_studio/docker-compose.yml`
- related 2026-04-28 tests.

## Validation Context

The orchestrator observed:
- targeted reviewer-finding tests -> `22 passed` and `11 passed`
- full suite -> `437 passed`
- Ruff -> pass
- Mypy -> pass
- build and Twine check -> pass

## Do Not

- Edit files.

## Output

Report blockers with exact file/line references, or state no blockers remain for these P1 findings.
