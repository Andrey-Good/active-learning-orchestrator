# PyTorch CUDA via uv

## Goal

Switch the project from a CPU-only PyTorch wheel to the official CUDA-enabled PyTorch wheel while keeping dependency management under `uv`.

## Steps

1. Inspect the current `pyproject.toml` dependency entry for `torch`.
2. Update `pyproject.toml` to use the official PyTorch CUDA wheel source.
3. Regenerate `uv.lock`.
4. Sync `.venv` from the updated lockfile.
5. Verify that the environment reports a CUDA-enabled torch build.

## Rules

- Use the official PyTorch wheel source, not an ad hoc manual install.
- Keep changes minimal and limited to dependency configuration plus generated lockfile.
- Do not change unrelated dependencies.
- Do not touch notebook code for this task.

## Notes

- PyTorch CUDA support on Windows is distributed via PyTorch-hosted wheels such as `cu128`.
- There is no separate generic `uv` flag that converts a CPU wheel into a CUDA wheel after the fact; the wheel source itself must be selected correctly.
