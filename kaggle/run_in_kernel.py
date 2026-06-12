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
# Clone outside /kaggle/working so the kernel OUTPUT (and log pulls) stay small — only our
# artifacts under OUT are downloaded, not the whole repo tree.
CLONE_DIR = "/tmp/al_repo"
OUT = "/kaggle/working/out"


def _resolve_preset() -> str:
    sibling = Path(__file__).resolve().parent / "al_preset.txt"
    if sibling.exists():
        value = sibling.read_text(encoding="utf-8").strip()
        if value:
            return value
    return os.environ.get("AL_PRESET", "mitigation")


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

    # Tee everything to OUT/run.log so the error is always pullable even when Kaggle's own
    # log API returns empty for a fast-failing kernel.
    log_path = Path(OUT) / "run.log"
    log = open(log_path, "w", encoding="utf-8")

    def runp(cmd: list[str], env: dict | None = None) -> int:
        log.write("$ " + " ".join(cmd) + "\n"); log.flush()
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:  # type: ignore[union-attr]
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return proc.wait()

    n_gpus = _gpu_count()
    header = f"[kernel] branch={BRANCH} gpus={n_gpus} preset={preset}\n"
    sys.stdout.write(header); log.write(header); log.flush()

    if n_gpus <= 1:
        rc = runp([sys.executable, bench, "--preset", preset, "--output-dir", OUT,
                   "--shard-index", "0", "--shard-count", "1"])
    else:
        procs = []
        for idx in range(n_gpus):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(idx)
            procs.append(subprocess.Popen(
                [sys.executable, bench, "--preset", preset, "--output-dir", OUT,
                 "--shard-index", str(idx), "--shard-count", str(n_gpus)], env=env))
        rc = 0 if all(p.wait() == 0 for p in procs) else 1
        runp([sys.executable, bench, "--merge-only", "--output-dir", OUT])

    if rc != 0:
        msg = f"[kernel] benchmark exited {rc}; running stats on whatever completed\n"
        sys.stdout.write(msg); log.write(msg); log.flush()
    runp([sys.executable, stats, "--input-dir", OUT])
    done = "[kernel] done. artifacts in " + OUT + "\n"
    sys.stdout.write(done); log.write(done); log.flush()
    log.close()


if __name__ == "__main__":
    main()
