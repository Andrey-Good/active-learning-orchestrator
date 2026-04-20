# TS Dataset Evaluation Plan

## Goal
- Add lightweight Hugging Face text-classification datasets to `lab/active_learning_lab.ipynb`.
- Run the TS notebook on them with a deliberately weak training budget so raw ECE is noticeably high before calibration.
- Report the measured metrics.

## Scope
- Update only the TS lab notebook and files under `plans/`.
- Keep the existing notebook structure intact.
- Use environment overrides for the actual runs instead of hardwiring all run settings into the notebook.

## Datasets to add
- `CogComp/trec`
- `cornell-movie-review-data/rotten_tomatoes`
- `SetFit/sst2`

## Steps
1. Inspect current dataset loader assumptions and confirm these datasets fit the normalized text/label schema.
2. Add dataset registry entries with conservative train/calibration/test slices.
3. Validate notebook JSON after edits.
4. Run quick experiments with a weak training budget to increase pre-calibration ECE.
5. Summarize raw vs calibrated metrics and note whether TS helps.

## Guardrails
- Do not change unrelated notebook logic.
- Prefer scalar calibration for evaluation unless the user asks otherwise.
- Keep runs lightweight and reproducible.
