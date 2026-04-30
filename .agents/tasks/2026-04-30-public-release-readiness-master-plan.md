# 2026-04-30 Public Release Readiness Master Plan

## Task Identifier

`release-readiness-master` - assess whether the current SDK worktree is suitable for a public GitHub release.

## Context

The user asked for a current release-readiness assessment for the SDK and a list of problems if any. The worktree is intentionally treated as the current candidate state. It is dirty and contains many uncommitted changes; do not revert or clean unrelated changes.

## Decomposition

1. Can this be split into independent parts without conflicts? Yes. Runtime/package behavior, repository hygiene/security, and validation-evidence review can be checked independently.
2. Which parts can run in parallel? The focused audit subtasks can run in parallel because they are read-only.
3. Where is strict sequencing required? Final synthesis must run after subtask findings and local verification results are available.

## Subtasks

- `rr01-blackbox-runtime-package`: black-box install, import, README quickstart, build artifact, and externally observable runtime behavior.
- `rr02-release-hygiene-security`: public GitHub hygiene, license, metadata, gitignore, secrets/leak risks, and release-blocking repository contents.
- `rr03-validation-evidence-review`: tests, lint/type/build/twine evidence, benchmark/report claims, and whether public claims match evidence.
- `rr04-final-system-review`: final cross-check after the above.

## Responsibility Boundaries

All subtasks are assessment-only. Do not implement fixes. Do not revert user changes. Do not stage, commit, push, or open a PR.

## In Scope

- Current worktree state.
- Public SDK release on GitHub as a beta/library release.
- Runtime behavior observable through package installation and public API use.
- Packaging metadata, docs, examples, tests, license, security posture, and claim safety.

## Out of Scope

- Fixing defects.
- Publishing to PyPI or GitHub.
- Changing product scope.
- Long-running uncapped benchmarks unless a short smoke is needed to validate a claim.

## Acceptance Criteria

- Final answer gives a clear release/no-release recommendation.
- Confirmed problems include severity, evidence, and practical release impact.
- Weak or speculative findings are either rejected or marked as residual risk.
- A concise report is saved under `.agents/tmp`.

## Expected Validations

- Inspect public docs/metadata and repository state.
- Run appropriate fast gates where feasible: import, quickstart/script, tests or focused tests, build, twine check, lint/type if time permits.
- Check packaged contents and public URLs.
- Review subagent findings critically before final synthesis.
