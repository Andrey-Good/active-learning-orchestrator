Task ID: W47
Short name: update reference report docs
Relation to overall task: Document the accepted reference benchmark conclusions so the user can see whether SDK heuristics are weaker than manual/external baselines.

Assumptions and resolved ambiguities:
- R60 accepted full reference benchmark artifacts.
- SDK formula strategies match manual formula macro-F1 AULC exactly overall and per dataset.
- Direct modAL/scikit-activeml runtime comparison is unavailable locally.
- BADGE/CoreSet are real missing method families, not drop-in probability-formula comparisons.
- This worker is not alone in the codebase. Do not revert or overwrite unrelated edits.

Goal and expected result:
- Update benchmark docs/report with the reference benchmark conclusion.
- Make the answer explicit: pure SDK entropy/margin/least-confidence are not weaker than manual formulas under identical probabilities; vanilla entropy underperformance is a known batch/data/model failure mode.
- Identify missing next methods: BADGE, CoreSet/k-center with embeddings, QBC/BALD where appropriate.

Responsibility boundaries:
- In scope:
  - `benchmarks/README.md`
  - `benchmarks/results/current_benchmark_report.md`
  - optional `benchmarks/results/reference_full/analysis.md` if only adding clarifying wording, not changing numbers
- Out of scope:
  - `src/**`
  - `tests/**`
  - root README unless absolutely needed
  - dependency manifests/lockfiles

Required numbers to include:
- Reference full: 495 metrics rows, 495 selection rows, 180 equivalence rows.
- Formula equivalence: mean Jaccard `0.990502`, min `0.729730`, exact order `146/180`.
- Macro-F1 AULC diffs SDK vs manual formula equivalents: `0.000000` overall/per dataset.
- Best overall in reference full: `mix_interleaved_class_group_random` mean macro-F1 AULC `0.997098`.

Acceptance criteria:
- Docs clearly distinguish manual formula parity from direct external-library runtime comparison.
- Docs do not claim BADGE/CoreSet are implemented.
- Docs say what would be needed to compare BADGE/CoreSet fairly.
- No code files are changed.

Expected validations:
- Check required numbers appear.
- Check no overclaim about direct modAL/skactiveml runtime benchmark.
