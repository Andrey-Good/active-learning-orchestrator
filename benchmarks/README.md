# SDK-First Benchmark Harness

This directory contains scriptable benchmarks for active-learning selection quality. The harness is intentionally not a notebook: it can be run from the repository root and writes small CSV/JSON/Markdown artifacts for comparison across SDK algorithm changes.

## Evidence Contract

Benchmark artifacts are claim-scoped evidence, not generic proof. The full contract is documented in [`docs/BENCHMARK_EVIDENCE.md`](../docs/BENCHMARK_EVIDENCE.md).

Newly generated manifests must include reproducibility metadata (`argv`, git SHA/dirty/status counts, Python/runtime/platform, artifact schema version), artifact names, `benchmark_claim_category`, and `benchmark_contract`.

SDK-first stochastic and committee rows produced by `sdk_first_benchmark.py` are proxy integration evidence only. The benchmark adapter creates deterministic perturbations around one scikit-learn model, so those rows do not prove true MC-dropout behavior or independently trained committee acquisition quality.

Supported claim categories are:

- `formula_parity`: SDK/manual formula equivalence under identical probability rows.
- `sdk_overhead`: tiny-fixture acquisition-only SDK/manual overhead evidence.
- `active_learning_quality`: diagnostic quality, coverage, stop-policy, and runtime behavior under stated datasets/budgets/seeds/adapters.
- `native_external_library_workflow_smoke`: opt-in native external query API smoke evidence.
- `end_to_end_public_project_workflow`: public `ActiveLearningProject` smoke workflow evidence.

Retained Stage 9 directories are accepted diagnostic evidence but may be legacy/pre-schema for manifest metadata. Formula shims named `modal_formula_*` and `skactiveml_formula_*` are not native external-library workflow benchmarks.

## Smoke Command

```powershell
uv run python benchmarks/sdk_first_benchmark.py --preset smoke
```

The default smoke preset runs two deterministic synthetic datasets, one seed, five strategy configs, and cumulative label budgets of `12,24,36`. Results are written to a timestamped directory such as `benchmarks/results/smoke/<run-id>/`.

Useful overrides:

```powershell
uv run python benchmarks/sdk_first_benchmark.py --preset full
uv run python benchmarks/sdk_first_benchmark.py --datasets rare_class_trap --strategies random,entropy,margin --budgets 16,32,48 --seeds 13,17
uv run python benchmarks/sdk_first_benchmark.py --preset smoke --output-dir benchmarks/results/local_smoke --overwrite
```

Explicit `--output-dir` paths are rejected when the directory is non-empty unless `--overwrite` is supplied. With `--overwrite`, the output directory is cleaned before current artifacts are written. Generated result directories are ignored by default; promote only reviewed evidence such as the retained Stage 9 directories or `current_benchmark_report.md`.

## Opt-In Real Dataset Commands

Real Hugging Face datasets are never included in the synthetic `smoke` or `full` defaults. They run only when named explicitly or through a real preset:

```powershell
uv run python benchmarks/sdk_first_benchmark.py --preset real_smoke
uv run python benchmarks/sdk_first_benchmark.py --preset real_medium --seeds 13,21,34 --max-train-samples 800 --max-test-samples 500
uv run python benchmarks/sdk_first_benchmark.py --preset real_full --seeds 13,21,34 --max-train-samples 1200 --max-test-samples 800
uv run python benchmarks/sdk_first_benchmark.py --preset real_smoke --datasets banking77 --strategies random,entropy --budgets 100,200
uv run python benchmarks/sdk_first_benchmark.py --preset real_full --datasets dair_ai_emotion --budgets 100 --seeds 13,21,34 --max-train-samples 300 --max-test-samples 300
```

Real presets use cumulative budgets of `100,200,300,500,800` by default. `real_smoke` is smoke-only local evidence and may use one seed for quick probes. `real_medium` and `real_full` are Stage 11 standard-real presets: they require at least three distinct seeds, for example `--seeds 13,21,34`, plus explicit positive `--max-train-samples` and `--max-test-samples` caps. `--allow-uncapped-real-standard` exists only for explicit local exploratory runs; artifacts produced with that override are marked `local_uncapped_override`, not standard evidence.

- `real_smoke`: `banking77`, with the small smoke strategy subset.
- `real_medium`: `banking77` and `clinc_oos_imbalanced`, with random, uncertainty, class/group-balanced entropy, and CoreSet coverage.
- `real_full`: all registered real datasets and all strategy names exposed by `strategy_specs()`.

Registered real datasets:

- `clinc_oos_imbalanced`: Hugging Face `clinc/clinc_oos`, config `imbalanced`, splits `train`, `validation`, `test`, text column `text`, label column `intent`.
- `clinc_oos_plus`: Hugging Face `clinc/clinc_oos`, config `plus`, splits `train`, `validation`, `test`, text column `text`, label column `intent`.
- `banking77`: Hugging Face `mteb/banking77`, config `default`, splits `train`, `test`, text column `text`, raw label column `label`, semantic label-name column `label_text`.
- `dair_ai_emotion`: Hugging Face `dair-ai/emotion`, config `split`, splits `train`, `validation`, `test`, text column `text`, label column `label`; intended as an easier sanity/coverage dataset.

The real loader requires the optional `datasets` package at runtime. Unit tests monkeypatch the loader and do not download from Hugging Face.

Use `--max-train-samples` and `--max-test-samples` for bounded all-methods probes. This is required for standard real evidence and important for O(n^2) diversity methods such as density-weighted diversity, which should not be run blindly on a full real pool without a cap.

Many-class real datasets do not require `--initial-seed-size` to be at least the number of labels. The benchmark only requires the chosen seed set to contain at least two classes so the scikit-learn classifier can fit. Synthetic datasets keep the stricter one-sample-per-class seed requirement.

`go_emotions_simplified` is intentionally deferred and is not a runnable default. GoEmotions is multi-label, so adding it requires an explicit single-label conversion or multi-label evaluation policy before fair comparison with the current single-label benchmark harness.

## Full Project Smoke Command

```powershell
uv run python benchmarks/sdk_first_benchmark.py --preset project_smoke
```

The project smoke drives the public `ActiveLearningProject` facade instead of only the scheduler-level API. It configures a project, imports an initial train seed through `project.import_labels(...)`, then verifies the first public `run_step(...)` performs the SDK seed train/eval before selection. It then executes one active public round through select, backend push, poll, pull, train/eval, and update. Results are written to a timestamped directory such as `benchmarks/results/project_smoke/<run-id>/`.

The benchmark-only oracle backend labels only samples that have been pushed to it, using a private label map that is not exposed through dataset fields. Its artifact records selected ids, completed rounds, status counts, project validation, acquisition-surface validation, and the exact public facade calls used.

## Reference Strategy Benchmark

```powershell
uv run python benchmarks/reference_strategy_benchmark.py --preset smoke
```

The reference harness compares SDK strategies with manual NumPy-style reference selectors under the same synthetic datasets, sklearn text model, pool order, seeds, budgets, and `SelectionContext.predict_proba(...)` outputs. It writes `metrics.csv`, `selections.csv`, `equivalence.csv`, `manifest.json`, `external_adapters.json`, `validation.json`, `summary.json`, and `summary.md` to timestamped directories such as `benchmarks/results/reference_smoke/<run-id>/` by default. Current newly generated manifests record argv, git SHA/dirty state, Python runtime, platform, artifact schema version, and artifact filenames. Some retained Stage 9 evidence directories are legacy/pre-schema artifacts and should not be cited for those manifest fields unless regenerated.

Manual references include `manual_random`, `manual_entropy`, `manual_margin`, `manual_least_confidence`, and `manual_class_group_balanced_entropy`. Formula-equivalent SDK/manual pairs are summarized with overlap counts, Jaccard, and exact-order diagnostics in `equivalence.csv`. `manual_random` remains a stochastic/hash baseline with a different deterministic contract than SDK random, so it is intentionally excluded from formula-equivalence diagnostics. Manual probability rows are strictly validated using the SDK uncertainty contract: row-like output, one row per pool id, consistent width, at least two finite non-negative probabilities, and each row must already sum to `1.0` within SDK tolerance. Count-like or logit-like rows are rejected instead of being normalized. Tie-breaking is deterministic and documented in the manifest.

Optional external-library formula hooks are defensive. `modal_formula_entropy`, `modal_formula_margin`, `modal_formula_uncertainty`, `skactiveml_formula_entropy`, `skactiveml_formula_margin`, and `skactiveml_formula_least_confidence` are benchmark-local formula shims and are skipped with clear reasons when `modAL` or `skactiveml` cannot be imported. They are not native external-library workflow benchmarks and do not measure query API behavior, setup cost, restart behavior, or end-to-end active-learning quality. The benchmark does not add those libraries as project dependencies.

Useful overrides:

```powershell
uv run python benchmarks/reference_strategy_benchmark.py --preset smoke --output-dir benchmarks/results/local_reference_smoke
uv run python benchmarks/reference_strategy_benchmark.py --preset smoke --output-dir benchmarks/results/local_reference_smoke --overwrite
uv run python benchmarks/reference_strategy_benchmark.py --preset full --datasets separable_topics,rare_class_trap --seeds 13,17
uv run python benchmarks/reference_strategy_benchmark.py --preset smoke --strategies entropy,manual_entropy,modal_formula_entropy
```

Reference benchmark output directories are rejected when non-empty unless `--overwrite` is supplied. With `--overwrite`, the output directory is cleaned before current artifacts are written. Generated result directories are local evidence by default; promote only reviewed benchmark artifacts.

## Manual Strategy Micro-Benchmark

```powershell
uv run python benchmarks/manual_strategy_benchmark.py --output-dir benchmarks/results/local_manual_strategy --overwrite
```

This small acquisition-only benchmark verifies that SDK uncertainty selections match hand-written formulas under frozen probability rows. It is useful for fast formula parity checks and SDK-overhead diagnostics, not for end-to-end model-quality claims.

## Native External Workflow Smoke

```powershell
uv run python benchmarks/native_external_benchmark.py --preset smoke
```

This separate opt-in smoke calls native external query APIs when optional libraries are installed. It writes `native_external_results.csv`, `native_external_summary.json`, `manifest.json`, and `summary.md` to a timestamped directory such as `benchmarks/results/native_external_smoke/<run-id>/`. Missing or incompatible optional libraries produce skipped rows with clear reasons instead of failing the whole run.

Useful overrides:

```powershell
uv run python benchmarks/native_external_benchmark.py --preset smoke --libraries modal
uv run python benchmarks/native_external_benchmark.py --preset smoke --strategies modal_native_entropy,skactiveml_native_uncertainty
uv run python benchmarks/native_external_benchmark.py --preset smoke --output-dir benchmarks/results/local_native_external --overwrite
```

Native rows include `modal_native_entropy`, `modal_native_margin`, `modal_native_least_confidence`, and `skactiveml_native_uncertainty`. These artifacts use claim category `native_external_library_workflow_smoke`; they are a native-query smoke, not a full end-to-end production comparison or performance-superiority claim. The benchmark does not add `modAL` or `skactiveml` as required project dependencies. Native external smoke support is separate from retained formula-shim/reference artifacts and should not be cited as Stage 11 retained quality evidence unless a promoted native-external artifact explicitly says so.

## Current Accepted Findings

The accepted diagnostic benchmark evidence is summarized in `benchmarks/results/current_benchmark_report.md`. The current promoted Stage 2C smoke is `benchmarks/results/stage2_smoke_current/`. The retained Stage 9 evidence directories are `benchmarks/results/stage9_final/` and `benchmarks/results/stage9_reference/`.

Headline conclusions:

- Stage 2C quality-gate parser check: `uv run pytest tests/test_quality_gate_report.py -q` -> `8 passed`.
- Stage 2C smoke command: `uv run python benchmarks/sdk_first_benchmark.py --preset smoke --output-dir benchmarks/results/stage2_smoke_current --overwrite` -> `30` metrics rows, `30` selection rows, and `30` stop-policy rows.
- Stage 2C quality gate command: `uv run python benchmarks/quality_gate_report.py benchmarks/results/stage2_smoke_current` -> PASS, schema version `2`, evidence category `sdk_native_synthetic_diagnostic`.
- Stage 2C passing strategies under the stricter combined gate are `least_confidence`, `margin`, and `mix_entropy_random`; `entropy`, `least_confidence`, `margin`, and `mix_entropy_random` all have at least one non-negative final macro-F1 or AULC lift vs matched random.
- The SDK-first Stage 9 run contains `1,440` metrics rows, `1,440` selection rows, and `864` stop-policy rows.
- The reference Stage 9 run contains `495` metrics rows, `495` selection rows, and `180` formula-equivalence rows.
- `class_group_balanced_entropy` is the best overall strategy in the retained diagnostic matrix with mean macro-F1 AULC `0.996018`.
- Random mean macro-F1 AULC is `0.948852`; the best overall delta versus random is `+0.047166`.
- Budget-16 macro-F1 is `0.986665` for the best strategy and `0.878105` for random.
- Conservative macro-F1 plateau stopping saves `30.21%` of labels on average with `-0.001866` mean quality delta.
- Formula-equivalence diagnostics have mean Jaccard `0.985537`, min Jaccard `0.684211`, and exact selected order in `139/180` rows. Equivalent probability-formula pairs have `0.000000` macro-F1 AULC diffs.
- Pure SDK entropy/margin/least-confidence matched the manual probability formulas under identical `predict_proba` outputs in the retained equivalence rows. Vanilla entropy underperformance in the diagnostic matrix should therefore be investigated as a batch/data/model failure mode rather than treated as formula-level evidence against the SDK implementation.
- The reference benchmark formula-shim rows do not validate direct `modAL` or `skactiveml` runtime behavior. Native external workflow smoke evidence belongs to the separate native external benchmark artifacts.
- BADGE, CoreSet/k-center with embeddings, stochastic/committee, hybrid, and mix SDK strategies are part of the Stage 9 SDK-first benchmark surface. Basic bandit scheduling exists in the SDK, but it was not part of the retained Stage 9 full preset. Optional external-library comparisons still require fair adapter-specific setup.
- These results are based on deterministic synthetic diagnostic datasets and a benchmark-only scikit-learn adapter. They should not be treated as proof of real-world production superiority.

## What Is Benchmarked

Datasets:

- `separable_topics`: balanced three-class text data with strong class-specific vocabulary.
- `rare_class_trap`: imbalanced three-class text data with a rare class that partially overlaps a majority class.
- `grouped_duplicates`: balanced labels with near-duplicate groups to expose duplicate and group-concentration behavior.
- Opt-in real datasets: `clinc_oos_imbalanced`, `clinc_oos_plus`, `banking77`, and `dair_ai_emotion`.

Strategies:

- `random`: SDK `RandomStrategy` through `StrategyScheduler(mode="single")`.
- `entropy`: SDK `EntropyStrategy` through `StrategyScheduler(mode="single")`.
- `group_diverse_entropy`: SDK `GroupDiverseEntropyStrategy` through `StrategyScheduler(mode="single")`.
- `class_balanced_entropy`: SDK class-balanced entropy strategy through `StrategyScheduler(mode="single")`.
- `class_group_balanced_entropy`: SDK class- and group-balanced entropy strategy through `StrategyScheduler(mode="single")`.
- `margin`: SDK `MarginStrategy` through `StrategyScheduler(mode="single")`.
- `least_confidence`: SDK `LeastConfidenceStrategy` through `StrategyScheduler(mode="single")`.
- `mix_entropy_random`: SDK scheduler mix with `{"entropy": 0.7, "random": 0.3}`.
- `mix_uncertainty_random`: SDK scheduler mix with `{"entropy": 0.4, "margin": 0.3, "random": 0.3}`.
- `mix_group_diverse_random`: SDK scheduler mix with `{"group_diverse_entropy": 0.4, "margin": 0.3, "random": 0.3}`.
- `mix_class_group_random`: SDK scheduler mix with `{"class_group_balanced_entropy": 0.7, "random": 0.3}`.
- `mix_class_group_margin_random`: SDK scheduler mix with `{"class_group_balanced_entropy": 0.4, "margin": 0.3, "random": 0.3}`.
- `mix_interleaved_class_group_random`: SDK interleaved scheduler mix with `{"class_group_balanced_entropy": 0.7, "random": 0.3}`.
- `mix_interleaved_class_group_margin_random`: SDK interleaved scheduler mix with `{"class_group_balanced_entropy": 0.4, "margin": 0.3, "random": 0.3}`.
- `coreset_kcenter`, `embedding_kmeans_pp`, `max_min_embedding`, `deduplicate_near_neighbors`, and `density_weighted_diversity`: SDK embedding/diversity strategies in the Stage 9 full preset.
- `badge`: SDK BADGE strategy in the Stage 9 full preset.
- stochastic and committee proxies: SDK strategies for stochastic prediction and disagreement-style acquisition in the Stage 9 full preset. These rows validate SDK integration and diagnostics with deterministic sklearn-model perturbations; they are not evidence of true MC-dropout or independently trained committee quality.
- hybrid rows: SDK combined strategies in the Stage 9 full preset. Basic adaptive-arm bandit scheduling is SDK functionality, but not part of this retained full benchmark preset.

Model:

- Benchmark-only scikit-learn adapter using TF-IDF features and logistic regression.
- The adapter exists only in `benchmarks/sdk_first_benchmark.py` and implements the SDK text model methods needed by `SelectionContext`.

## Metrics And Artifacts

Each run emits:

- `metrics.csv`: budgeted accuracy, balanced accuracy, macro-F1, weighted-F1, macro recall, rare-class recall when applicable, calibration metrics, zero-recall class counts/fractions, label/class coverage counts/fractions, missing-label/class counts/fractions, missing test-support-weighted fraction, newly selected label counts/fractions, normalized AULC, lift versus the random baseline, runtime, and per-label budget efficiency metrics.
- `selections.csv`: selected ids, scheduler snapshots, selected and cumulative label distributions, duplicate selected counts, selected group counts, top-group fraction, and group HHI concentration.
- `stop_policies.csv`: post-hoc plateau stop-policy decisions for each dataset/strategy/seed curve, including stopped budget, full budget, labels saved, relative savings, stop/full metric values, quality delta, and runtime saved.
- `full_train_reference.csv`: full-supervised no-budget reference metrics, including calibration metrics, from fitting the benchmark model on the full train split for each dataset/seed.
- `manifest.json`: run configuration, stop-policy definitions, artifact names, argv, git SHA/dirty state, Python version, platform, artifact schema version, plus the current SDK API gap for newly generated SDK-first benchmark artifacts. Retained Stage 9 artifacts are legacy/pre-schema where those fields are absent.
- `summary.json`: machine-readable rollup of best macro-F1 rows and stop-policy rows.
- `summary.md`: compact human-readable summary with stop-policy diagnostics.
- `validation.json`: acquisition-surface checks proving that labels are absent from visible ids, groups, schema, and metadata, and that sorted ids/groups do not form label blocks.

Inside each project-smoke output directory, the project smoke emits:

- `summary.json`: strict JSON proof of public seed train/eval, active project-loop execution, imported seed labels, selected ids, completed round details, backend push/pull ids, validation status, and final counts.
- `summary.md`: compact human-readable project-loop summary.
- `manifest.json`: reproducibility metadata and artifact names.
- `workdir/state.json`: SDK project state produced by the public facade.

JSON artifacts are written as strict JSON. Metrics that are not applicable to a dataset, such as rare-class recall for datasets without a rare class, are represented as `null` in JSON rollups.

Calibration columns are computed from test-set `predict_proba(...)` rows aligned to `dataset.labels` after every fit:

- `multiclass_brier_score`: mean squared distance between the predicted probability vector and the one-hot true label.
- `nll`: mean negative log-likelihood of the true label probability, clipped only to avoid `log(0)`.
- `ece`: expected calibration error using confidence bins over max-probability predictions.

## SDK Usage And Current Gap

Acquisition is SDK-first: the harness calls `active_learning_sdk.engine.StrategyScheduler.select_batch(...)` with a `SelectionContext`, so random, entropy, margin, least-confidence, and scheduler mix behavior come from the SDK rather than benchmark-local strategy formulas.

The `SelectionContext` provider used during acquisition exposes sample text, split metadata, and opaque group ids only. Acquisition-visible `sample_id` and `group_id` values are assigned after deterministic split-level shuffles so sorted ids and group ids do not expose label-ordered ranges. True labels remain in benchmark-private state for seed construction, oracle labeling after selection, training, evaluation, and diagnostics, but they are not present in acquisition-visible `DataSample.meta`.

The scheduler smoke intentionally remains scheduler-level because it compares strategy curves cheaply across datasets and budgets. The separate `project_smoke` preset covers the full public project state machine now that `ActiveLearningProject.import_labels(...)` exists. It does not externally warm-start the model, mutate private engine state, or create fake completed rounds.

## Determinism

Synthetic data generation, initial seed selection, model training, and scheduler calls are seeded. For fixed CLI arguments and dependency versions, artifact contents should be reproducible except for run timestamps and elapsed runtime values.
