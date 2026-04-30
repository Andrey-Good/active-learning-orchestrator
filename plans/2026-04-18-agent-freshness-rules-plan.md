# AGENT Freshness Rules Plan

## Goal

Add explicit freshness and precedence rules to `AGENT.md` so an automated sync agent does not overwrite newer and more accurate guidance in `AGENT.md` with older content from `CLAUDE.md`.

## Scope

- Update only `AGENT.md`.
- Keep the rules concise and operational.
- Preserve the existing "CLAUDE sync" logic while clarifying source priority and recency handling.

## Steps

1. Define source priority between user instructions, `AGENT.md`, codebase reality, and `CLAUDE.md`.
2. Add rules for conflict resolution and recency checks.
3. Make it explicit that `CLAUDE.md` is an input source, not an authority over `AGENT.md`.
4. Keep the final wording short enough for repeated automated use.

## Validation

- The sync agent can distinguish authority from reference material.
- `AGENT.md` cannot be downgraded by older `CLAUDE.md` content.
- The rules stay short and clear.
