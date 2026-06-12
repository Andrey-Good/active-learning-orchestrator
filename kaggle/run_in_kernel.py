"""
Kaggle kernel bootstrap for the transformer active-learning benchmark.

This single self-contained script is the kernel's ``code_file``. It does NOT bundle the SDK:
it ``git clone``s the public repo (a chosen branch), then runs the benchmark + statistics
from that clone. Nothing is uploaded from a local machine — the kernel pulls the public code.

Requirements (set in kernel-metadata.json): GPU accelerator ON, Internet ON. Torch ships in the
Kaggle base image; we only pip-install transformers + datasets if missing.

Outputs land in /kaggle/working/out: metrics.csv, alc_summary.csv, statistical_tests.csv,
bootstrap_ci.csv, learning_curves.png.

Configure the repo/branch/preset via the constants below (push_and_run.py rewrites BRANCH and
writes al_preset.txt next to this file).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/Andrey-Good/active-learning-orchestrator.git"
BRANCH = "benchmark/transformer-al-pilot"
CLONE_DIR = "/kaggle/working/repo"
OUT = "/kaggle/working/out"


def _resolve_preset() -> str:
    sibling = Path(__file__).resolve().parent / "al_preset.txt"
    if sibling.exists():
        value = sibling.read_text(encoding="utf-8").strip()
        if value:
            return value
    return os.environ.get("AL_PRESET", "deadline")


def _gpu_count() -> int:
    try:
        import torch  # type: ignore

        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


def _ensure_deps() -> None:
    missing = []
    for mod in ("transformers", "datasets", "sklearn", "scipy", "matplotlib"):
        try:
            __import__(mod)
        except Exception:
            missing.append("scikit-learn" if mod == "sklearn" else mod)
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=False)


def _clone() -> Path:
    target = Path(CLONE_DIR)
    if not target.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(target)],
            check=True,
        )
    return target


def main() -> None:
    preset = _resolve_preset()
    code_dir = _clone()
    bench = str(code_dir / "benchmarks" / "transformer_benchmark.py")
    stats = str(code_dir / "benchmarks" / "statistical_analysis.py")
    Path(OUT).mkdir(parents=True, exist_ok=True)
    _ensure_deps()

    n_gpus = _gpu_count()
    print(f"[kernel] branch={BRANCH} gpus={n_gpus} preset={preset}", flush=True)

    if n_gpus <= 1:
        subprocess.run(
            [sys.executable, bench, "--preset", preset, "--output-dir", OUT,
             "--shard-index", "0", "--shard-count", "1"],
            check=True,
        )
    else:
        procs = []
        for idx in range(n_gpus):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(idx)
            procs.append(
                subprocess.Popen(
                    [sys.executable, bench, "--preset", preset, "--output-dir", OUT,
                     "--shard-index", str(idx), "--shard-count", str(n_gpus)],
                    env=env,
                )
            )
        if any(p.wait() != 0 for p in procs):
            print("[kernel] WARNING: a shard exited non-zero; merging what completed", flush=True)
        subprocess.run([sys.executable, bench, "--merge-only", "--output-dir", OUT], check=True)

    subprocess.run([sys.executable, stats, "--input-dir", OUT], check=True)
    print("[kernel] done. artifacts in", OUT, flush=True)


if __name__ == "__main__":
    main()
