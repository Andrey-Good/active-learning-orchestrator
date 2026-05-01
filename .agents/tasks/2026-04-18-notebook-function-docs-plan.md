# Notebook Function Documentation Plan

## Goal

Add brief, high-signal documentation to notebook functions and methods so it is easier to understand:

- what inputs they expect;
- the structure of dictionaries, dataframes, and sequences passed in;
- what they return;
- the important part of the internal logic when that affects usage.

## Scope

- Update only `active_learning_lab.ipynb`.
- Do not change runtime behavior.
- Keep documentation concise and practical.

## Execution Order

1. Add docstrings to dataset utility functions.
2. Add docstrings to model wrapper methods.
3. Add docstrings to selection/oracle backend helpers.
4. Add docstrings to runner/result functions.

## Rules

- Prefer short docstrings over long comments.
- Explain nested structures only when they are not obvious.
- Mention dataframe columns explicitly when a function depends on them.
- Mention return shapes/types when they matter for downstream code.
- Do not add noisy commentary to self-explanatory lines.
