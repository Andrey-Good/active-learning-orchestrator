# Notebook Implementation Plan

## Goal

Create one clean experiment notebook for this repository that:
- generates a configurable synthetic text dataset;
- loads selected real datasets from Hugging Face with fingerprint verification;
- supports a small local training pipeline and an SDK-based pipeline;
- supports choosing between a lightweight model and `FacebookAI/roberta-base`;
- saves run results to a compact table next to the notebook;
- is easy to copy and modify for future experiments;
- runs successfully in the current environment.

## Constraints

- Do not modify repository files other than:
  - files inside `plans/`;
  - one new notebook file;
  - Python environment packages installed via `uv`.
- Keep the notebook readable and intentionally minimal.
- Avoid unnecessary settings and excessive metrics.
- The final deliverables outside `plans/` must be only:
  - installed libraries in the environment;
  - one notebook file.

## Assumptions

- The notebook may use the repository source directly via `src/`.
- Results table may be written at runtime next to the notebook based on current working directory.
- Notebook validation can use a smoke configuration with the lightweight model and a small synthetic dataset.
- SDK mode will use a notebook-local oracle labeling backend instead of the incomplete Label Studio scaffold.

## Execution Order

1. Inspect the minimum SDK/runtime contracts required by the notebook.
2. Identify the minimal dependency set.
3. Install those dependencies with `uv` into the existing environment.
4. Implement one notebook with:
   - compact configuration;
   - dataset registry and synthetic generator;
   - fingerprint helpers;
   - model wrappers;
   - local runner;
   - SDK runner;
   - result persistence.
5. Run the notebook in a smoke configuration.
6. Fix only issues inside the notebook until the smoke run succeeds.

## Stages

### Stage 1. Dependency setup

Install only what is needed for:
- notebook execution;
- Hugging Face datasets/models;
- training;
- result table output;
- notebook smoke validation.

### Stage 2. Notebook implementation

Build the notebook as one self-contained file with clear sections:
- overview and config;
- synthetic dataset generation;
- Hugging Face dataset import and hash verification;
- dataset selection;
- model selection;
- local training path;
- SDK training path;
- result saving.

### Stage 3. Validation

Run the notebook end-to-end with a cheap smoke setup and verify:
- dataset generation works;
- local and/or selected default pipeline works;
- results table is produced;
- notebook exits without execution errors.

## Must Do

- Keep code clean and senior-reviewable.
- Prefer simple functions/classes over over-engineering.
- Make the notebook easy to copy for new experiments.
- Use clear defaults that are cheap to execute.
- Keep saved results compact and comparable.

## Must Not Do

- Do not edit SDK source files.
- Do not add helper modules or extra scripts.
- Do not add unnecessary plotting/metrics/config sprawl.
- Do not install broad dependency bundles without need.
- Do not rely on the unfinished Label Studio integration.

## Important Risks And Checks

- `roberta-base` is expensive; notebook defaults must remain lightweight.
- Real dataset schemas differ; normalization must be explicit per dataset.
- Fingerprint verification must be stable enough for repeated runs.
- SDK mode must avoid dependence on unimplemented backends.
- Notebook execution validation must use a bounded smoke config to avoid long runs.

## Acceptance Criteria

- One notebook file exists and is readable.
- Required libraries are installed in the environment via `uv`.
- The notebook runs successfully in smoke mode.
- No files outside `plans/`, the notebook, and environment packages are modified.
