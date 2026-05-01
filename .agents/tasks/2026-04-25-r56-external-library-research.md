Task ID: R56
Short name: external active-learning library research
Relation to overall task: Establish what competitor libraries actually implement for uncertainty and batch/diversity methods before benchmarking against them.

Goal and expected result:
- Research primary sources for `modAL`, `scikit-activeml`, and at least one BADGE/core-set reference implementation or paper/source.
- Determine which comparisons are feasible in this repo without adding permanent dependencies.
- Produce a concise research note under `docs/` summarizing formulas, API behavior, source links, and benchmark recommendations.

Responsibility boundaries:
- In scope:
  - `docs/reference_active_learning_libraries.md`
  - optional read-only shell checks for package import availability
- Out of scope:
  - modifying `src/**`
  - modifying `benchmarks/**`
  - modifying dependency manifests or lockfiles
  - installing permanent dependencies

Important constraints:
- Use primary sources where possible: official docs, GitHub source, papers.
- If external packages cannot be imported, document why and recommend optional comparison hooks instead of forcing dependency changes.
- Keep citations/links in the doc.

Execution plan:
- Inspect official docs/source for modAL uncertainty sampling.
- Inspect scikit-activeml strategy overview/API and uncertainty sampling behavior.
- Inspect BADGE/core-set references enough to decide feasibility for this SDK.
- Check whether `skactiveml` or `modAL` imports in current env without modifying lockfiles.
- Write findings and benchmark recommendations.

Acceptance criteria:
- Research note states whether SDK pure entropy/margin/least-confidence should match manual/external formulas on identical probabilities.
- Research note identifies differences likely caused by batch construction, tie-breaking, calibration, model, or diversity.
- No source/benchmark files are edited.

Expected validations:
- Link/source checks.
- Optional `uv run python -c "..."` import checks.
