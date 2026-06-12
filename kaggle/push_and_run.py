"""
Local driver: push the Kaggle GPU kernel that runs the transformer AL benchmark, poll it to
completion, and pull the result artifacts.

The kernel itself (``run_in_kernel.py``) ``git clone``s the public repo branch and runs from
there, so this driver uploads only the small bootstrap script — no source-tree upload, no dataset.

Credentials: the global ``~/.kaggle/access_token`` (the new CLI's cached token) shadows
everything, so to act as a second account, redirect HOME for this process to a dir holding that
account's ``kaggle.json``. Example (panampalmers, leaving gordeevmax untouched):

    $env:USERPROFILE="C:\\Users\\gorde\\kgl_home_pana"; $env:HOME=$env:USERPROFILE
    python kaggle/push_and_run.py --preset smoke

Usage:
    python kaggle/push_and_run.py --username <user> --preset deadline
    python kaggle/push_and_run.py --preset smoke --no-wait
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KERNEL_SLUG = "al-transformer-benchmark"


def _resolve_username(explicit: str | None) -> str:
    if explicit:
        return explicit
    config_dir = os.environ.get("KAGGLE_CONFIG_DIR") or str(Path.home() / ".kaggle")
    cfg = Path(config_dir) / "kaggle.json"
    if cfg.exists():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        if data.get("username"):
            return str(data["username"])
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    raise SystemExit("Could not resolve Kaggle username; pass --username or set up kaggle.json.")


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    print("  $", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=False, text=True, capture_output=True)


def push_kernel(kaggle: str, username: str, preset: str) -> str:
    kdir = REPO / "kaggle" / "_build" / "kernel"
    if kdir.exists():
        shutil.rmtree(kdir)
    kdir.mkdir(parents=True)
    shutil.copy2(REPO / "kaggle" / "run_in_kernel.py", kdir / "run_in_kernel.py")
    (kdir / "al_preset.txt").write_text(preset, encoding="utf-8")
    kernel_id = f"{username}/{KERNEL_SLUG}"
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_id,
        "title": "AL Transformer Benchmark",
        "code_file": "run_in_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2), encoding="utf-8")
    res = run([kaggle, "kernels", "push", "-p", str(kdir)])
    print((res.stdout or "") + (res.stderr or ""))
    return kernel_id


def wait_and_pull(kaggle: str, kernel_id: str, poll_seconds: int, results_dir: Path) -> None:
    print(f"[poll] watching {kernel_id} every {poll_seconds}s ...")
    terminal = ("complete", "error", "cancel")
    while True:
        status = run([kaggle, "kernels", "status", kernel_id])
        text = ((status.stdout or "") + (status.stderr or "")).strip()
        print("  status:", text)
        if any(t in text.lower() for t in terminal):
            break
        time.sleep(poll_seconds)
    results_dir.mkdir(parents=True, exist_ok=True)
    run([kaggle, "kernels", "output", kernel_id, "-p", str(results_dir)])
    print(f"[done] artifacts -> {results_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", default=None)
    parser.add_argument("--kaggle-bin", default="kaggle")
    parser.add_argument("--preset", default="deadline")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--results-dir", default=str(REPO / "kaggle" / "results"))
    args = parser.parse_args()

    username = _resolve_username(args.username)
    print(f"[auth] kaggle username: {username}")
    # On Windows the CLI is a .bat shim; subprocess needs its full resolved path.
    kaggle_bin = shutil.which(args.kaggle_bin) or args.kaggle_bin
    kernel_id = push_kernel(kaggle_bin, username, args.preset)
    if args.no_wait:
        print(f"pushed. watch: https://www.kaggle.com/code/{kernel_id}")
        return
    wait_and_pull(kaggle_bin, kernel_id, args.poll_seconds, Path(args.results_dir))


if __name__ == "__main__":
    main()
