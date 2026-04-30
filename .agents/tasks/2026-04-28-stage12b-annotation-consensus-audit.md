# Stage 12B: Annotation Policy And Consensus Audit

## Task Identifier

stage12b-annotation-consensus-audit

## Context

Professional human-labeling workflows need safe aggregation, conflict handling,
and review paths. Existing code has annotation policies and timeout behavior, but
Stage 12 asks for consensus/conflict workflow hardening.

## Goal

Audit annotation policy behavior and identify P1/P2 gaps for Stage 12.

## Responsibility Boundaries

In scope:

- `src/active_learning_sdk/annotation.py`
- annotation-related configs and types
- engine pull/timeout behavior
- tests around annotation policy, needs-review, multi-annotator behavior
- docs mentioning consensus/conflicts

Out of scope:

- Editing code.
- Backend transport details except how annotations feed policy.

## Files May Be Changed

- Only `.agents/tmp/2026-04-28-stage12b-annotation-consensus-audit.md`

## Review Questions

1. Does `AnnotationPolicy` support latest/majority/consensus/conflict behavior
   honestly and safely?
2. Does `allow_single_annotator=False` work?
3. Are disagreements routed to `needs_review` with labels cleared?
4. Are timeouts safe and auditable?
5. Are multi-label policies explicitly supported or rejected?
6. Are tests covering edge cases?

## Expected Output

Severity-ranked audit with recommended Stage 12 implementation scope.

## Forbidden Actions

- No code/docs/test edits.
