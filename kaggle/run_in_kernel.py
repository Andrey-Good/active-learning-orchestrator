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

# Sentinel constants injected at push time by push_and_run.py.
# push_and_run.py replaces "= None" with "= <value>" so per-kernel config is baked into
# the uploaded code file — Kaggle script kernels only upload the code file, not sidecar .txt
# files, so sidecar-based config silently reverts to default. These sentinels fix that.
_PRESET_OVERRIDE = None  # e.g. "v2_phase0"
_PROTOCOLS_OVERRIDE = None  # e.g. "cold"


def _resolve_preset() -> str:
    # Priority: baked override > sidecar txt > env > default.
    if _PRESET_OVERRIDE is not None:
        return str(_PRESET_OVERRIDE)
    sibling = Path(__file__).resolve().parent / "al_preset.txt"
    if sibling.exists():
        value = sibling.read_text(encoding="utf-8").strip()
        if value:
            return value
    return os.environ.get("AL_PRESET", "v2_phase0")


def _resolve_protocols() -> list[str]:
    """Which training protocols to run, in order. Default: both. A baked ``_PROTOCOLS_OVERRIDE``
    (or sibling ``al_protocols.txt``, or ``AL_PROTOCOLS`` env) with e.g. ``cold`` lets us split
    cold/warm across two CONCURRENT kernels (each using both T4s) to halve wall-clock with no
    quality cost.  Priority: baked override > sidecar txt > env > default."""
    # Priority: baked override > sidecar txt > env > default.
    raw = ""
    if _PROTOCOLS_OVERRIDE is not None:
        raw = str(_PROTOCOLS_OVERRIDE)
    if not raw:
        sibling = Path(__file__).resolve().parent / "al_protocols.txt"
        if sibling.exists():
            raw = sibling.read_text(encoding="utf-8").strip()
    if not raw:
        raw = os.environ.get("AL_PROTOCOLS", "cold,warm")
    protocols = [p.strip() for p in raw.split(",") if p.strip() in ("cold", "warm")]
    return protocols or ["cold", "warm"]


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
    protocols = _resolve_protocols()
    header = f"[kernel] branch={BRANCH} gpus={n_gpus} preset={preset} protocols={protocols}\n"
    sys.stdout.write(header); log.write(header); log.flush()

    # Run the selected protocol(s) into the same OUT dir.
    # The protocol column in each row distinguishes passes.
    # Each protocol gets its own per-protocol shard files; --merge-only combines all of them
    # into a single metrics.csv after all passes complete.
    overall_rc = 0
    for protocol in protocols:
        proto_header = f"[kernel] === protocol={protocol} ===\n"
        sys.stdout.write(proto_header); log.write(proto_header); log.flush()
        if n_gpus <= 1:
            rc = runp([sys.executable, bench, "--preset", preset, "--output-dir", OUT,
                       "--protocol", protocol, "--shard-index", "0", "--shard-count", "1"])
        else:
            # Each shard captures its own stdout+stderr to a per-shard log so tracebacks are
            # never lost.  Shards still run in parallel (no serialization).
            # On any non-zero exit, the tail of that shard's log is appended to run.log so a
            # single pulled run.log shows the error without needing to fetch extra files.
            procs = []
            shard_logs = []
            for idx in range(n_gpus):
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(idx)
                shard_log_path = Path(OUT) / f"shard_{protocol}_{idx}.log"
                shard_logs.append(shard_log_path)
                shard_log_fh = open(shard_log_path, "w", encoding="utf-8")
                procs.append((subprocess.Popen(
                    [sys.executable, bench, "--preset", preset, "--output-dir", OUT,
                     "--protocol", protocol,
                     "--shard-index", str(idx), "--shard-count", str(n_gpus)],
                    env=env,
                    stdout=shard_log_fh,
                    stderr=subprocess.STDOUT,
                ), shard_log_fh))
            rc = 0
            for idx, (p, fh) in enumerate(procs):
                exit_code = p.wait()
                fh.close()
                shard_log_path = shard_logs[idx]
                # Stream shard output to console for live Kaggle log visibility.
                shard_content = shard_log_path.read_text(encoding="utf-8", errors="replace")
                sys.stdout.write(shard_content)
                log.write(shard_content)
                log.flush()
                if exit_code != 0:
                    rc = 1
                    # Append shard tail to run.log so the error is immediately visible.
                    tail_lines = shard_content.splitlines()[-80:]
                    tail_text = "\n".join(tail_lines)
                    err_msg = (
                        f"\n[kernel] === shard {protocol}_{idx} FAILED (exit {exit_code}) — tail ===\n"
                        f"{tail_text}\n"
                        f"[kernel] === end shard {protocol}_{idx} tail ===\n"
                    )
                    sys.stdout.write(err_msg)
                    log.write(err_msg)
                    log.flush()
        if rc != 0:
            msg = f"[kernel] {protocol} pass exited {rc}; continuing to next protocol\n"
            sys.stdout.write(msg); log.write(msg); log.flush()
            overall_rc = rc

    # Merge all protocol + shard files into one canonical metrics.csv.
    runp([sys.executable, bench, "--merge-only", "--output-dir", OUT])

    if overall_rc != 0:
        msg = f"[kernel] one or more passes failed; running stats on whatever completed\n"
        sys.stdout.write(msg); log.write(msg); log.flush()
    runp([sys.executable, stats, "--input-dir", OUT])
    done = "[kernel] done. artifacts in " + OUT + "\n"
    sys.stdout.write(done); log.write(done); log.flush()
    log.close()


if __name__ == "__main__":
    main()
