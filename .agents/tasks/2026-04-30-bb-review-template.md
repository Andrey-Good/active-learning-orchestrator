# Black-Box Finding Reviewer Template

## Goal

Review one worker report for false positives, weak evidence, misuse of public APIs, environment-only failures, and missing reproduction detail.

## Allowed Context

- The worker task document and worker-owned evidence directory.
- Public docs and package metadata used by the worker.
- Runtime reproduction commands, if needed.

## Must Not Touch

- `src/**`
- `tests/**`
- Product source files.

## Review Questions

- Is each suspected issue reproducible from public docs or public API behavior?
- Could the issue be caused by incorrect usage, an unrealistic assumption, or bad local setup?
- Is expected behavior documented or reasonably implied by a public contract?
- Is actual behavior evidenced by logs/artifacts?
- Are severity and confidence calibrated?
- Is there a useful regression-test recommendation?

## Output

Write a reviewer report in the matching worker evidence directory as `review.md`.
