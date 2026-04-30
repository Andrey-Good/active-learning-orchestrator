# Baseline Notebook Parameter Plan

## Goal

Adjust the notebook defaults so a plain run produces a reasonable baseline result instead of a weak smoke-test result.

## Changes

1. Increase training budget:
   - more active-learning rounds;
   - larger batch sizes;
   - more train epochs.
2. Improve the default synthetic dataset:
   - larger train and test splits;
   - lower label noise.
3. Use defaults that are cheaper to understand and less fragile:
   - local mode by default;
   - lightweight model by default;
   - safer default acquisition strategy.

## Rules

- Keep the notebook simple.
- Prefer config changes over logic rewrites.
- Do not add unnecessary metrics or features.
