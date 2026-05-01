Task ID: PLAN-2026-04-25
Short name: reference active learning study
Relation to overall task: New research phase requested by the user to honestly compare SDK heuristics against manual logic and external active-learning libraries, then improve SDK if it underperforms.

Objective:
- Determine whether SDK uncertainty/diversity heuristics are intrinsically weak, incorrectly implemented, or simply failing on known batch active-learning failure modes.
- Compare SDK selections and learning curves against manual reference implementations and external library behavior where feasible.
- Use benchmark evidence to guide any further SDK changes.

Task boundaries:
- In scope:
  - Research external AL libraries and source/documentation.
  - Add reference benchmark code and artifacts.
  - Add tests for reference/manual parity and any new SDK improvements.
  - Update benchmark docs/report with honest conclusions.
- Out of scope for this plan unless evidence requires it:
  - Rewriting the whole SDK.
  - Adding heavy deep-learning BADGE training if it cannot fit the current fast benchmark constraints.
  - Claiming real-world superiority from synthetic-only results.

Decomposition:
- R56: external competitor/source research, read-only plus optional doc output.
- W44: implement reference benchmark harness for SDK vs manual and optional external libraries.
- R57: review reference benchmark implementation.
- W45: run reference benchmark and write analysis.
- R58: review reference benchmark artifacts/analysis.
- Later conditional tasks:
  - If SDK differs from manual/external on same formulas, fix the SDK implementation.
  - If SDK matches but vanilla heuristics underperform, add/test stronger methods such as density-weighted uncertainty, k-center embeddings, BADGE-lite, or QBC.
  - If external libraries expose stronger feasible methods, implement comparable SDK variants.

Parallelism:
- R56 can run in parallel with local feasibility checks.
- W44 should use R56 findings where possible, but can start with manual parity first.
- Implementation and artifact reviews are sequential.

Completion criteria:
- A reproducible reference benchmark exists.
- It reports SDK-vs-manual parity for entropy/margin/least-confidence/random-like baselines.
- External library comparison is included if feasible without destabilizing project dependencies.
- If SDK is worse under equivalent formulas, the cause is diagnosed and fixed.
- If SDK is not worse under equivalent formulas, the report says so clearly and identifies the next real algorithmic gap.
