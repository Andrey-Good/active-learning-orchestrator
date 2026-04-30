Task ID: W38
Short name: benchmark hybrid mix probe
Relation to overall task: Test whether the accepted `class_group_balanced_entropy` strategy can be improved at early budgets and rare-class recall by mixing it with complementary arms.

Assumptions and resolved ambiguities:
- R47 accepted `class_group_balanced_entropy` benchmark results.
- The hybrid is best on mean macro-F1 AULC and group concentration, but early budget and rare-class recall are not uniformly best.
- Existing mix scheduler can combine any registered strategy by name.
- This worker is not alone in the codebase. Do not revert or overwrite unrelated edits.

Goal and expected result:
- Add benchmark-only strategy specs for one or two hybrid mixes.
- Run a targeted comparative benchmark across all diagnostic datasets.
- Produce result artifacts and an honest `analysis.md` describing whether hybrid mixes improve early macro-F1 or rare recall without regressing primary macro-F1 AULC/group concentration.

Responsibility boundaries:
- In scope:
  - `benchmarks/sdk_first_benchmark.py`
  - `benchmarks/README.md` if documentation needs updating
  - `benchmarks/results/hybrid_mix_probe/`
- Out of scope:
  - `src/**`
  - `tests/**`
  - root README
  - Docker/backends

Architectural constraints and forbidden actions:
- Do not change SDK implementation.
- Do not weaken validation/leakage checks.
- Do not overwrite existing result directories.
- Keep comparison deterministic and comparable to W37.

Suggested strategy specs:
- `mix_class_group_random`: `SchedulerConfig(mode="mix", mix={"class_group_balanced_entropy": 0.7, "random": 0.3})`
- `mix_class_group_margin_random`: `SchedulerConfig(mode="mix", mix={"class_group_balanced_entropy": 0.4, "margin": 0.3, "random": 0.3})`

High-level execution plan:
- Add the suggested strategy specs, unless a better minimal pair is justified in `analysis.md`.
- Run:
  `uv run python benchmarks/sdk_first_benchmark.py --datasets grouped_duplicates,rare_class_trap,separable_topics --strategies random,entropy,group_diverse_entropy,class_balanced_entropy,class_group_balanced_entropy,mix_group_diverse_random,mix_class_group_random,mix_class_group_margin_random --seeds 13,17,23 --budgets 16,32,48,64,96 --output-dir benchmarks/results/hybrid_mix_probe`
- Validate strict JSON and expected row counts: 3 datasets x 8 strategies x 3 seeds x 5 budgets = 360 rows for metrics and selections.
- Write/ensure `analysis.md` with primary and guardrail conclusions.

Step -> verify:
- Strategy specs added -> verify CLI accepts them.
- Benchmark run -> verify artifacts exist.
- Validation -> verify strict JSON parse and row counts.
- Analysis -> verify reported aggregates match CSV/summary data.

Acceptance criteria:
- Artifacts exist under `benchmarks/results/hybrid_mix_probe/`.
- Analysis states whether either mix beats the single hybrid on early metrics or rare recall and whether it preserves macro-F1 AULC/group concentration.
- Failed hypotheses must be reported honestly.

Expected tests and validations:
- `python -m py_compile benchmarks/sdk_first_benchmark.py`
- Benchmark command above
- Strict JSON parse
- CSV row count validation

Dependencies:
- Sequentially after R47.

Parallel/sequential notes:
- Benchmark-only task. Can run in parallel with a read-only scheduler audit.
