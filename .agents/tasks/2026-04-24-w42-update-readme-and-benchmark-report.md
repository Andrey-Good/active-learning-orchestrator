Task ID: W42
Short name: update README and benchmark report
Relation to overall task: Final documentation update after removing old notebooks, replacing the benchmark system, and improving SDK strategies/scheduler based on measured results.

Assumptions and resolved ambiguities:
- Old notebooks and legacy benchmark CSVs were removed.
- New benchmark harness lives at `benchmarks/sdk_first_benchmark.py`.
- Accepted current best strategy/scheduler findings:
  - `class_group_balanced_entropy` is the best single strategy in the diagnostic matrix.
  - `mix_interleaved_class_group_random` is the best balanced mix candidate from the final probe: better rare recall and slightly better AULC than the single hybrid, with the same group concentration guardrail, but slightly worse budget-16 macro-F1.
  - `mix_class_group_margin_random` gives higher early/primary metrics but regresses group concentration and should not be promoted as the safest default.
- This worker is not alone in the codebase. Do not revert or overwrite unrelated edits.

Goal and expected result:
- Update root `README.md` so it describes the current product surface and current benchmark evidence, not removed notebooks or obsolete artifacts.
- Add or update a concise benchmark report file if useful, preferably under `benchmarks/results/`, summarizing accepted benchmark findings and recommended defaults.

Responsibility boundaries:
- In scope:
  - `README.md`
  - `benchmarks/README.md`
  - optional consolidated report under `benchmarks/results/`
- Out of scope:
  - `src/**`
  - `tests/**`
  - benchmark CSV/JSON result artifacts unless only linking/describing them
  - pyproject/lockfiles
  - Docker/backends implementation

Architectural constraints and forbidden actions:
- Do not claim production readiness beyond verified evidence.
- Do not mention deleted notebooks as current entrypoints.
- Do not invent benchmark numbers; use existing accepted analysis files.
- Keep README practical: install, minimal usage, strategies, Label Studio/Docker, tests, benchmarks, limitations.
- Mention synthetic benchmark external-validity limits.

Required benchmark numbers to include:
- Full tests: `uv run --group dev pytest -q` -> `42 passed`.
- From `benchmarks/results/class_group_balanced_entropy/analysis.md`:
  - hybrid `class_group_balanced_entropy` mean macro-F1 AULC: grouped duplicates `0.997088`, rare class trap `0.993787`, separable topics `0.999382`.
  - group HHI/top-group fraction: `0.072321` across diagnostic datasets.
- From `benchmarks/results/mix_interleaved_probe/analysis.md`:
  - `mix_interleaved_class_group_random`: macro-F1 AULC `0.997098`, budget-16 macro-F1 `0.976866`, rare recall AULC `0.990000`, top-group fraction `0.072321`, group HHI `0.072321`.
  - Compared with block `mix_class_group_random`: macro-F1 AULC `+0.002176`, budget-16 macro-F1 `+0.018879`, rare recall AULC `+0.001667`, top-group fraction `-0.038591`, group HHI `-0.007331`.
  - `mix_class_group_margin_random` improves quality but regresses group concentration.

High-level execution plan:
- Read current README and benchmark README.
- Replace stale notebook/legacy benchmark sections.
- Document current strategies:
  - `random`
  - `entropy`
  - `margin`
  - `least_confidence`
  - `group_diverse_entropy`
  - `class_balanced_entropy`
  - `class_group_balanced_entropy`
  - `coreset_kcenter` placeholder
- Document scheduler modes:
  - `single`
  - `mix`
  - `mix_interleaved`
  - `custom`
  - `bandit` placeholder
- Add commands for smoke/full/project-smoke benchmarks.
- Add concise benchmark report or summary table.
- Ensure old notebook paths/legacy CSV claims are removed.

Step -> verify:
- README updated -> verify stale notebook names no longer appear as active entrypoints.
- Benchmark report updated/created -> verify numbers match accepted analysis files.
- Links/commands checked -> verify referenced files exist.

Acceptance criteria:
- README no longer documents deleted notebooks or old benchmark CSVs as current.
- README includes current strategy/scheduler names and current benchmark conclusions.
- Optional consolidated report is easy to cite from final response.
- No code/test files changed by this task.

Expected validations:
- Search README/benchmark docs for stale notebook references.
- Basic file existence checks for linked benchmark analysis files.

Dependencies:
- Sequentially after R52.

Parallel/sequential notes:
- Documentation-only task. Must be independently reviewed.
