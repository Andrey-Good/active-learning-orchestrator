# W83 - Stage 9 README And Roadmap Update

## Context

Stage 9 release audit found stale public README claims:

- reports are described as unimplemented even though JSON/Markdown/HTML reports now work;
- `coreset_kcenter` is described as a placeholder even though embedding diversity strategies are implemented;
- annotation timeouts are described as unenforced even though the WAIT step enforces them;
- strategy lists omit BADGE, embedding, stochastic/committee, hybrid, and bandit support;
- test count is stale at `47 passed` instead of the current `234 passed`;
- benchmark report has been refreshed under `benchmarks/results/current_benchmark_report.md`.

## Goal

Update public documentation so it accurately describes the current SDK as a usable beta/product-quality active-learning SDK, while keeping limitations honest.

## Responsibility Boundaries

You own documentation updates only.

## In Scope

- `README.md`
- `docs/SDK_REAL_PRODUCT_ROADMAP.md`
- `benchmarks/README.md` only if needed for consistency

## Out of Scope

- Any `src/**` runtime code
- Any `tests/**`
- `pyproject.toml`
- Benchmark reruns

## Required Content

Root README must include:

- Current product surface.
- Minimal usage example.
- Strategy list:
  - `random`, `entropy`, `margin`, `least_confidence`
  - `group_diverse_entropy`, `class_balanced_entropy`, `class_group_balanced_entropy`
  - `coreset_kcenter`, `embedding_kmeans_pp`, `max_min_embedding`, `deduplicate_near_neighbors`, `density_weighted_diversity`
  - `badge`
  - stochastic/committee strategies
  - hybrid/mix modes
- Label Studio external and managed Docker setup.
- Simulator backend.
- Reports: strict JSON summary, Markdown, HTML, manifest, returned artifact paths.
- Stop criteria and annotation timeout behavior.
- Fresh Stage 9 benchmark headlines from `benchmarks/results/current_benchmark_report.md`:
  - SDK final row counts: 1,440 metrics, 1,440 selections, 864 stop-policy rows.
  - Reference row counts: 495 metrics, 495 selections, 180 equivalence rows.
  - Best overall mean macro-F1 AULC: `class_group_balanced_entropy` `0.996018`.
  - Random mean macro-F1 AULC: `0.948852`; delta `+0.047166`.
  - Budget-16 macro-F1: best `0.986665`, random `0.878105`.
  - Conservative macro-F1 plateau: `30.21%` mean relative label savings, `-0.001866` quality delta.
  - Formula equivalence: mean Jaccard `0.985537`, min Jaccard `0.684211`, exact-order `139/180`, AULC diffs `0.000000`.
- Current test result: `uv run --group dev pytest -q` -> `234 passed`.
- Honest limitations:
  - synthetic benchmark evidence only;
  - optional `modAL`/`skactiveml` runtime comparison skipped locally;
  - Hugging Face adapter still limited/scaffold unless user supplies real training implementation;
  - LLM backend remains placeholder;
  - bandit is basic, not a mature production optimizer;
  - no broad real-world dataset benchmark yet.

Roadmap must be updated from stale "everything missing" framing to "completed foundation + remaining professional gaps".

## Style

- Clear, concise, technically honest.
- Do not claim universal superiority or production proof.
- Prefer exact command snippets and artifact paths.
- Keep README useful for a new user.

## Expected Validation

- Read the final docs for stale contradictions listed above.
- Run no code unless needed for docs verification.

## Dependencies

Depends on W82 benchmark report and A82 release audit.
