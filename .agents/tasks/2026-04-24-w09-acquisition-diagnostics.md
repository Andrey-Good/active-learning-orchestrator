# W09 - Acquisition Diagnostics

## Relation to Overall Task
This is the next scientific optimization step after baseline sweep metadata is fixed and reviewed. Current evidence suggests uncertainty heuristics can underperform random; before changing methods, we need diagnostics explaining why.

## Assumptions and Resolved Ambiguities
- Do not launch this task until W08 is reviewed cleanly.
- Use existing strategies only: random, entropy, margin, least-confidence.
- Diagnostics should explain behavior; they should not alter strategy behavior yet.
- Active learning is budget-limited, so diagnostics must be per-budget/per-round.

## Goal
Add compact diagnostics to the learning-curve benchmark artifacts to inspect selected batches and acquisition behavior.

## Responsibility Boundaries
- Own benchmark runner/artifacts and optional learning-curve notebook documentation.
- Do not change SDK source or strategy logic.

## In Scope
- Add a diagnostics mode or always-on compact artifact for sweep/default runs.
- Record selected batch diagnostics such as:
  - per-round selected label distribution;
  - cumulative selected label distribution;
  - duplicate selected ID checks;
  - selected index/order distribution if available;
  - overlap between strategies for same dataset/model/seed/budget;
  - optional score summary/tie-rate for uncertainty methods if cheap.
- Generate CSV/JSON/Markdown artifacts under `benchmarks/results/learning_curves/`.
- Summarize whether heuristics are class-skewed or redundant vs random.

## Out of Scope
- No changes to acquisition algorithms.
- No new strategies.
- No SDK source changes.
- No long real-dataset benchmark.

## Files/Modules May Change
- `benchmarks/run_learning_curve_experiments.py`
- `lab/learning_curve_lab.ipynb` if documenting the diagnostics command
- New/updated diagnostics artifacts under `benchmarks/results/learning_curves/`

## Files/Areas Must Not Touch
- `src/active_learning_sdk/**`
- `README.md`
- Docker files

## Architectural Constraints and Forbidden Actions
- Diagnostics must be reproducible from raw benchmark rows and selected IDs.
- Do not infer true labels from unavailable data; use payload frame label maps available during the run.
- Keep artifacts reviewable in size.

## Step -> Verify Plan
- Add selected-label diagnostics generation -> verify counts sum to `new_labels`/`labeled_rows`.
- Add overlap diagnostics -> verify pairwise overlaps are bounded by batch sizes.
- Generate diagnostics for accepted baseline candidate(s) and key failing controls -> verify artifacts exist.
- Summarize findings -> verify they support or reject concrete next hypotheses.

## Acceptance Criteria
- Artifacts make it easy to see why random/heuristics differ.
- Diagnostics include `sweep_candidate`, strategy, seed, budget/round.
- Final report identifies at least one next testable hypothesis, such as tie-break/cold-start/warm-start/calibration.

## Expected Tests and Validations
- py_compile runner.
- Diagnostics artifact integrity script.
- Existing sweep command still works or diagnostics command works.

## Dependencies
- Must wait for W08 and its review.

## Parallel/Sequential Notes
- Sequential after W08 review.
