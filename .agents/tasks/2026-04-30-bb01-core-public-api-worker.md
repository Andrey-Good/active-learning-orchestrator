# BB01 Core Public API Worker

## Goal

Black-box stress test the documented root API, quickstart, public facade, model adapter capability contract, and scheduler/strategy behavior.

## Allowed Context

- Public docs: `README.md`, `docs/README.md`, `docs/SDK_CONTRACTS.md`.
- Package metadata: `pyproject.toml`.
- Runtime behavior through `uv run python`, installed package imports, generated workdirs, logs, and command output.

## Write Scope

- `.agents/tmp/blackbox/bb01-core-api/**`

## Must Not Touch

- `src/**`
- `tests/**`
- Existing benchmark result directories except read-only artifact behavior through documented commands.
- Any source files outside the write scope.

## In Scope

- Validate README simulator quickstart.
- Verify stable root exports import cleanly and optional adapters are lazy enough for root import.
- Exercise invalid public configurations and classify exceptions.
- Stress adapter contract violations for `fit`, `evaluate`, `predict_proba`, stochastic/committee shapes, invalid probability rows, and missing capabilities.
- Probe documented strategies and scheduler modes using small external fixtures.
- Check batch size and exhausted-pool behavior.

## Out Of Scope

- Report generation and cache artifact details, unless needed to reproduce core API behavior.
- Benchmark harness claims.
- Label Studio live service behavior.

## Acceptance Criteria

- Produce a concise worker report in `.agents/tmp/blackbox/bb01-core-api/report.md`.
- Include reproduction scripts/commands and captured outputs/logs for every suspected issue.
- Separate confirmed findings from rejected or inconclusive suspicions.

## Expected Validation

- Use standalone scripts created in the write scope.
- Prefer deterministic synthetic fixtures.
- Keep all claims grounded in observed command output.
