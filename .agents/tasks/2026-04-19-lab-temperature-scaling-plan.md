# Lab Temperature Scaling Plan

## Goal

Add a professional but compact `TemperatureScaling` implementation to the copied lab notebook.

## Scope

- Work only in the copied notebook under `lab/`.
- Keep the main model training unchanged.
- Fit temperature scaling only after model training is complete.
- Apply calibrated logits only during post-training inference/evaluation.

## Required Public API

1. `fit(...)`
   - Find temperature `T` from calibration logits and labels.
   - Use `LBFGS`.
2. `transform(...)`
   - Apply temperature scaling to logits.
3. `calibration_metrics(...)`
   - Return several probability-quality metrics.

## Design Rules

- Keep `T > 0` via parameterization, not by ad hoc clamping.
- Work on logits, not on already-softmaxed probabilities.
- Keep class independent from model training code.
- Use a small, explicit integration point after training.

## Integration Plan

1. Add `calibration` split handling if already prepared in config.
2. Add the `TemperatureScaling` class near model/calibration code.
3. Fit it in `run_local_experiment(...)` only after training is done.
4. Apply it when producing post-training probabilities/logits for evaluation.
