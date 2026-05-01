# AGENT And CLAUDE Sync Plan

## Goal

Update `AGENT.md` so it contains only the useful, stable guidance that should be carried over from `CLAUDE.md`, plus clear rules for future daily sync checks.

## Scope

- Read `AGENT.md` and `CLAUDE.md`.
- Add only high-value repository invariants and implementation-status guidance.
- Add concise rules that help an automated daily agent decide whether anything from `CLAUDE.md` belongs in `AGENT.md`.
- Do not modify `CLAUDE.md`.

## Constraints

- Keep `AGENT.md` concise.
- Avoid copying broad architecture walkthroughs, PRD ideas, or temporary planning material.
- Remove duplication rather than appending noise.

## Steps

1. Identify stable repository rules and invariants from `CLAUDE.md`.
2. Exclude descriptive, speculative, and low-signal content.
3. Add concise repo-specific guidance to `AGENT.md`.
4. Add explicit sync rules for a daily automated agent.
5. Keep the final file compact and operational.

## Must Include

- Public API and layering invariants.
- Resume/idempotency invariants.
- Strategy and adapter constraints that affect implementation.
- High-level current implementation/scaffold status.
- Rules for deciding what should and should not be synchronized from `CLAUDE.md`.

## Must Exclude

- Long architectural explanations.
- Future roadmap and PRD target features.
- Historical notes and broad project descriptions.
- Anything already better kept in guides or README-style docs.

## Validation

- `AGENT.md` becomes more useful for future work.
- `AGENT.md` stays short and free of clutter.
- `CLAUDE.md` remains unchanged.
