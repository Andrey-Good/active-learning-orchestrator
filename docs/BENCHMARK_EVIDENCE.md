# Benchmark Evidence Contract

Benchmark artifacts are evidence for specific claims only. A generated `manifest.json` must state the artifact schema version, argv, git SHA and dirty/status count, Python/runtime/platform details, artifact names, `benchmark_claim_category`, and `benchmark_contract`.

## Claim Categories

| Category | Current support | What it proves | What it does not prove |
| --- | --- | --- | --- |
| `formula_parity` | `benchmarks/reference_strategy_benchmark.py`, retained `benchmarks/results/stage9_reference/`, and focused formula microbenchmarks listed in `README.md` | SDK uncertainty formulas can match manual probability formulas under the same pool, model, `predict_proba` rows, seeds, budgets, and deterministic tie-breaking contract. | Native external-library workflow parity, external library query API behavior, or real-world superiority. |
| `sdk_overhead` | Retained W97/W98 acceptance reference microbenchmarks listed in `README.md` | Tiny-fixture acquisition-only SDK/manual overhead under a fixed local workload. | End-to-end runtime, service latency, large-pool scaling, or production overhead. |
| `active_learning_quality` | `benchmarks/sdk_first_benchmark.py`, retained `benchmarks/results/stage9_final/`, local synthetic gates, and capped real-data gates listed in `README.md` | Diagnostic lift, AULC, coverage, stop-policy, and runtime behavior versus matched random baselines under the configured synthetic or capped real-data fixtures. | Broad uncapped real-world dominance, scientific generalization across domains/model families, or native external-library workflow comparison. |
| `native_external_library_workflow_smoke` | `benchmarks/native_external_benchmark.py` Stage 2B smoke artifacts. | Optional native external query APIs can be called, or missing/incompatible libraries are recorded as skipped rows with reasons, under a reproducible smoke manifest. | Full end-to-end external-library quality comparison, setup-cost comparison, restart behavior, or performance superiority. |
| `end_to_end_public_project_workflow` | `--preset project_smoke` output from `benchmarks/sdk_first_benchmark.py` | Public `ActiveLearningProject` facade wiring for seed import and one active round through select, backend push, wait, pull, train/eval, and update. | External human-labeling service behavior or active-learning quality superiority. |

## Artifact Status

Newly generated SDK-first, reference, and project-smoke manifests are schema-bearing artifacts. They should be preferred when citing reproducibility metadata because they record the claim category and benchmark contract directly.

Retained Stage 9 directories are accepted diagnostic evidence, but some artifacts are legacy/pre-schema. Treat missing manifest fields in retained Stage 9 JSON as historical artifact limitations, not as permission to omit those fields from new runs.

Retained capped-real Banking77 evidence with two seeds is diagnostic only. It is not Stage 11 standard real evidence because standard real reports now require at least three seeds, explicit train/test caps, and calibration metrics in both `metrics.csv` and `full_train_reference.csv`.

## Stage 11 Standard Real Reports

Standard real reports use `benchmarks/sdk_first_benchmark.py --preset real_medium` or `--preset real_full`. They must include explicit positive `--max-train-samples` and `--max-test-samples` caps and at least three distinct seeds:

```powershell
uv run python benchmarks/sdk_first_benchmark.py --preset real_medium --seeds 13,21,34 --max-train-samples 800 --max-test-samples 500
uv run python benchmarks/sdk_first_benchmark.py --preset real_full --seeds 13,21,34 --max-train-samples 1200 --max-test-samples 800
```

`real_smoke` remains available for local probes, but its manifest is marked `smoke_only`. `--allow-uncapped-real-standard` is only for explicit local exploratory runs and marks output as `local_uncapped_override`; it should not be promoted as standard evidence.

Calibration columns are part of the Stage 11 benchmark evidence contract:

- `multiclass_brier_score`: mean squared distance between predicted probabilities and the one-hot true label.
- `nll`: mean negative log-likelihood of the true label probability.
- `ece`: expected calibration error from confidence bins over max-probability predictions.

The quality gate report summarizes calibration and fails standard real reports when required calibration, label/class coverage, zero-recall, seed-count, or full-train calibration evidence is missing.

For active-learning quality claims, one non-random strategy must show non-negative final-budget macro-F1 lift, non-negative AULC lift, all-budget non-loss rate of at least two thirds, final-budget non-loss rate of at least two thirds, and at least one positive final-budget or AULC lift versus matched random. The report also checks selection differentiation: all non-random strategies must not collapse to the same selected order, and no pair of non-random strategies may produce identical selected orders across every comparable dataset/seed/budget group.

## Citation Rules

- Cite `formula_parity` only for SDK/manual formula equivalence under identical probability rows.
- Cite `active_learning_quality` only with the dataset, budget, seed, adapter, and cap constraints that produced the result.
- Cite `sdk_overhead` only for acquisition-only tiny-fixture overhead unless a newer benchmark explicitly broadens the contract.
- Do not cite formula shims as native `modAL` or `skactiveml` workflow evidence.
- Keep native external command support separate from retained artifact evidence. Native external smoke artifacts prove only opt-in native query API reachability unless a promoted artifact explicitly broadens that claim.
- Do not claim production or universal superiority from deterministic synthetic datasets or capped real-data smoke gates.
