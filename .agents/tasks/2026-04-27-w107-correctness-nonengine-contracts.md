# Task W107: Non-Engine Correctness Contracts

## Context

The all-objections backlog includes serious non-engine correctness items around annotation aggregation, label schema validation, DataFrame/export JSON safety, Label Studio compatibility, and adapter/backend scaffolds.

## Goal

Improve production correctness in non-engine files without touching `engine.py`.

## Ownership

You may edit:

- `src/active_learning_sdk/annotation.py`
- `src/active_learning_sdk/configs.py`
- `src/active_learning_sdk/dataset/provider.py`
- `src/active_learning_sdk/backends/label_studio.py`
- `src/active_learning_sdk/backends/base.py`
- adapter files if needed
- tests that directly cover these areas

Do not edit:

- `src/active_learning_sdk/engine.py`
- strategy modules
- README/docs

## Required Fixes

1. `AnnotationPolicy.allow_single_annotator=False` must require distinct annotators for `min_votes`.
2. Majority ties should route to review unless a tie policy exists.
3. `LabelSchema.validate()` must reject non-string labels, empty labels, duplicates, and unhashable values via `ConfigurationError`.
4. `DataFrameDatasetProvider` should normalize extra scalar values into JSON-safe values where practical.
5. Label Studio existing project binding must not silently mutate incompatible label config when `project_id` or same title is reused.

## Constraints

- Keep behavior backward-compatible where it is not unsafe.
- Add focused tests for changed contracts.
- Do not edit engine/export code.

## Acceptance Criteria

- Add or update tests for annotation distinct annotators/ties and label schema validation.
- Run focused tests and report results.
