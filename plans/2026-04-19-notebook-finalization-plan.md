# Notebook Finalization Plan

## Goal

Bring `active_learning_lab.ipynb` to a final baseline-ready state that is:

- deterministic for identical code and parameters;
- easier to configure from the second code cell;
- easier to navigate by reading the code directly.

## Scope

- Update only `active_learning_lab.ipynb`.
- Do not add new files outside `plans/`.
- Keep the notebook compact and practical.

## Work Stages

### 1. Determinism

- Strengthen `seed_everything(...)`:
  - Python RNG
  - NumPy RNG
  - Torch RNG
  - CUDA RNG
  - cuDNN deterministic flags
  - optional deterministic algorithm mode
- Add an experiment-level seed helper so each `(dataset, model, mode, strategy)` run is reproducible.
- Use deterministic generators where the code relies on shuffled loaders or random sampling.

### 2. Config Cell Cleanup

- Turn the second code cell into clearly separated blocks:
  - run metadata
  - experiment controls
  - training hyperparameters
  - storage paths
  - synthetic dataset config
  - dataset registry
  - model registry
- Keep defaults explicit and readable.
- Preserve env override support, but make it secondary to the human-readable config layout.

### 3. Inline Navigation Comments

- Add short comments only where the pattern is not obvious:
  - `sys.path` injection for local SDK imports
  - class-id mapping
  - token mixing in synthetic generation
  - Hugging Face row normalization
  - random-vs-uncertainty batch selection
  - margin sorting trick
  - oracle task-id mapping
  - dataframe moves between labeled/unlabeled pools
  - SDK workdir reset and explicit split wiring

### 4. Keep Baseline Behavior Clean

- Do not add noisy features.
- Do not add large new abstractions.
- Keep return contracts unchanged so the rest of the notebook still reads simply.

## Rules

- Prefer small high-signal comments over dense prose.
- Prefer explicit parameter names over implicit behavior.
- Keep determinism implementation strict enough for repeated notebook runs on the same machine.
- Avoid changing the notebook's external usage pattern unless it improves clarity directly.
