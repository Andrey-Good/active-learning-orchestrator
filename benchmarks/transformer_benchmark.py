"""
Transformer active-learning benchmark (honest, real-model pilot).

Unlike ``sdk_first_benchmark.py`` (TF-IDF + LogisticRegression, with faked
committee/MC-dropout/BADGE proxies), this harness runs the *real* SDK acquisition strategies
against a *real* fine-tuned DistilBERT (`DistilBERTALAdapter`) on *real* text-classification
datasets, and writes per-round learning-curve metrics for downstream significance analysis.

Design:
- Drives the genuine ``SelectionContext`` + ``StrategyScheduler`` from the SDK, so the
  strategies under test are exactly the production code paths.
- Cold-restart fine-tune per round (fair AL protocol).
- 2-GPU sharding via ``--shard-index/--shard-count`` (run one process per GPU, each pinned with
  ``CUDA_VISIBLE_DEVICES``); per-curve resume via a checkpoint file.
- Optimized for small AL pools (fp16 autocast lives in the adapter; pools are capped).

Presets:
- ``smoke``    : tiny in-memory synthetic data, no network/HF-datasets needed (validates the
                 full code path + adapter on a laptop GPU/CPU).
- ``deadline`` : TREC + AG News via HF ``datasets``, 6 real strategies, seeds [13,21,34],
                 budgets [50,100,200,400] — sized for ~1h on Kaggle T4x2.

Example (single process / local smoke):
    python benchmarks/transformer_benchmark.py --preset smoke --output-dir runs/smoke

Example (Kaggle T4x2, two shells):
    CUDA_VISIBLE_DEVICES=0 python benchmarks/transformer_benchmark.py --preset deadline \
        --output-dir /kaggle/working/out --shard-index 0 --shard-count 2
    CUDA_VISIBLE_DEVICES=1 python benchmarks/transformer_benchmark.py --preset deadline \
        --output-dir /kaggle/working/out --shard-index 1 --shard-count 2
    python benchmarks/transformer_benchmark.py --merge-only --output-dir /kaggle/working/out
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# Ensure ``src`` is importable when run directly from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from active_learning_sdk import (  # noqa: E402
    DataSample,
    LabelSchema,
    SchedulerConfig,
    SelectionContext,
    StrategyScheduler,
)


def _make_adapter(kind: str, labels: Sequence[str], *, seed: int, model_name: str, device: str | None) -> Any:
    """Adapter factory. ``distilbert`` = real transformer; ``fake`` = numpy stub for offline smoke."""
    if kind == "distilbert":
        from active_learning_sdk.adapters.transformer import DistilBERTALAdapter

        return DistilBERTALAdapter(labels, model_name=model_name, seed=seed, device=device)
    if kind == "fake":
        return HashingNearestCentroidAdapter(labels, seed=seed)
    raise ValueError(f"unknown adapter kind: {kind}")


class HashingNearestCentroidAdapter:
    """
    Lightweight numpy-only adapter for OFFLINE smoke testing of the benchmark harness.

    It is NOT a transformer and produces no scientific result — it exists so the full pipeline
    (AL loop, real SDK strategies, SelectionContext wiring, CSV + stats) can be validated without
    torch/transformers installed. It implements the same contract as ``DistilBERTALAdapter``:
    fit / evaluate / predict_proba / embed / gradient_embed / get_model_id. Learning is a hashing
    bag-of-words + nearest-centroid classifier, so curves improve with more labels.
    """

    _DIM = 256

    def __init__(self, labels: Sequence[str], *, seed: int = 13) -> None:
        import numpy as np  # noqa: F401

        self.labels = [str(label) for label in labels]
        self._label_to_id = {label: i for i, label in enumerate(self.labels)}
        self.seed = int(seed)
        self._centroids: Any = None
        self._round = 0

    def _featurize(self, texts: Sequence[str]) -> Any:
        import numpy as np

        mat = np.zeros((len(texts), self._DIM), dtype=np.float64)
        for r, text in enumerate(texts):
            for token in str(text).lower().split():
                mat[r, hash(("tok", token)) % self._DIM] += 1.0
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return mat / norms

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs: Any) -> None:
        import numpy as np

        feats = self._featurize(texts)
        ids = np.array([self._label_to_id[str(label)] for label in labels])
        centroids = np.zeros((len(self.labels), self._DIM), dtype=np.float64)
        for c in range(len(self.labels)):
            mask = ids == c
            if mask.any():
                centroids[c] = feats[mask].mean(axis=0)
        self._centroids = centroids
        self._round += 1

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        import numpy as np

        feats = self._featurize(texts)
        if self._centroids is None:
            uniform = [1.0 / len(self.labels)] * len(self.labels)
            return [list(uniform) for _ in texts]
        logits = feats @ self._centroids.T * 8.0
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return [[float(v) for v in row] for row in probs]

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

        true = [str(label) for label in labels]
        proba = self.predict_proba(texts)
        pred = [self.labels[max(range(len(row)), key=row.__getitem__)] for row in proba]
        return {
            "accuracy": float(accuracy_score(true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
            "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
            "macro_recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        }

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [[float(v) for v in row] for row in self._featurize(texts)]

    def gradient_embed(self, texts: Sequence[str], labels: Any | None = None, batch_size: int = 32) -> list[list[float]]:
        import numpy as np

        feats = self._featurize(texts)
        probs = np.asarray(self.predict_proba(texts))
        pseudo = probs.argmax(axis=1)
        out: list[list[float]] = []
        for i in range(len(texts)):
            residual = probs[i].copy()
            residual[pseudo[i]] -= 1.0
            out.append([float(v) for v in np.outer(residual, feats[i]).reshape(-1)])
        return out

    def get_model_id(self) -> str:
        return f"fake-nc-r{self._round}-s{self.seed}"


# --------------------------------------------------------------------------- data model


@dataclass(frozen=True)
class Sample:
    sample_id: str
    text: str
    label: str
    split: str


@dataclass(frozen=True)
class Dataset:
    name: str
    labels: list[str]
    samples: list[Sample]


class Provider:
    """Minimal dataset provider satisfying ``SelectionContext``'s needs."""

    def __init__(self, samples: Sequence[Sample]) -> None:
        self._by_id = {s.sample_id: s for s in samples}

    def iter_sample_ids(self) -> Iterable[str]:
        return iter(self._by_id.keys())

    def get_sample(self, sample_id: str) -> DataSample:
        s = self._by_id[str(sample_id)]
        return DataSample(sample_id=s.sample_id, data={"text": s.text}, meta={"split": s.split})

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        return [self.get_sample(sid) for sid in sample_ids]

    def get_texts(self, sample_ids: Sequence[str]) -> list[str]:
        return [self._by_id[str(sid)].text for sid in sample_ids]

    def schema(self) -> dict[str, str]:
        return {"sample_id": "str", "text": "str", "split": "str"}


# --------------------------------------------------------------------------- datasets

# name -> (hf_path, hf_config, text_column, label_column, train_split, test_split)
# NOTE: use Parquet-backed mirrors (SetFit org) — Kaggle's current `datasets` library no longer
# supports legacy script-based datasets (e.g. bare "trec"/"ag_news" raise "Dataset scripts are no
# longer supported"). SetFit datasets expose columns: text, label (int), label_text.
_HF_DATASETS: dict[str, tuple[str, str | None, str, str, str, str]] = {
    "ag_news": ("SetFit/ag_news", None, "text", "label", "train", "test"),
    "sst2": ("SetFit/sst2", None, "text", "label", "train", "validation"),
    "emotion": ("SetFit/emotion", None, "text", "label", "train", "test"),
}

_DATASET_SEED = 12345  # fixes the (capped) pool so it is identical across AL seeds


def _cap_indices(n: int, cap: int | None) -> list[int]:
    indices = list(range(n))
    if cap is not None and cap < n:
        rng = random.Random(_DATASET_SEED)
        rng.shuffle(indices)
        indices = sorted(indices[:cap])
    return indices


def load_hf_dataset(name: str, max_train: int | None, max_test: int | None) -> Dataset:
    from datasets import load_dataset  # type: ignore

    hf_path, hf_config, text_col, label_col, train_split, test_split = _HF_DATASETS[name]
    ds = load_dataset(hf_path, hf_config) if hf_config else load_dataset(hf_path)
    train_raw = ds[train_split]
    test_raw = ds[test_split]

    def build(raw: Any, split: str, cap: int | None) -> list[Sample]:
        texts = raw[text_col]
        labels = raw[label_col]
        out: list[Sample] = []
        for i in _cap_indices(len(texts), cap):
            out.append(
                Sample(
                    sample_id=f"{name}-{split}-{i}",
                    text=str(texts[i]),
                    label=str(int(labels[i])),
                    split=split,
                )
            )
        return out

    train = build(train_raw, "train", max_train)
    test = build(test_raw, "test", max_test)
    labels = sorted({s.label for s in train} | {s.label for s in test}, key=lambda x: int(x))
    return Dataset(name=name, labels=labels, samples=train + test)


def build_synthetic_dataset(name: str = "synthetic") -> Dataset:
    """Tiny separable 3-class dataset for offline smoke tests (no HF datasets needed)."""
    vocab = {
        "sports": ["match", "goal", "team", "score", "coach", "league"],
        "science": ["atom", "experiment", "theory", "data", "physics", "lab"],
        "finance": ["invoice", "market", "stock", "refund", "account", "loan"],
    }
    labels = sorted(vocab.keys())
    rng = random.Random(7)
    samples: list[Sample] = []
    for split, per_class in (("train", 40), ("test", 16)):
        for label, terms in vocab.items():
            for j in range(per_class):
                words = [rng.choice(terms) for _ in range(6)] + [rng.choice(["the", "a", "of"]) for _ in range(3)]
                rng.shuffle(words)
                samples.append(
                    Sample(sample_id=f"{name}-{split}-{label}-{j}", text=" ".join(words), label=label, split=split)
                )
    return Dataset(name=name, labels=labels, samples=samples)


def load_dataset_by_name(name: str, max_train: int | None, max_test: int | None) -> Dataset:
    if name == "synthetic":
        return build_synthetic_dataset()
    return load_hf_dataset(name, max_train, max_test)


# --------------------------------------------------------------------------- AL loop


def choose_initial_seed(train: Sequence[Sample], labels: Sequence[str], size: int, seed: int) -> list[str]:
    rng = random.Random(seed + 404)
    by_label: dict[str, list[str]] = defaultdict(list)
    for s in train:
        by_label[s.label].append(s.sample_id)
    for ids in by_label.values():
        ids.sort()
    seed_ids: list[str] = []
    for label in labels:
        ids = by_label.get(label, [])
        if ids:
            seed_ids.append(rng.choice(ids))
    chosen = set(seed_ids)
    pool = [s.sample_id for s in train if s.sample_id not in chosen]
    rng.shuffle(pool)
    while len(seed_ids) < size and pool:
        seed_ids.append(pool.pop())
    return seed_ids


def run_one_curve(
    dataset: Dataset,
    strategy_name: str,
    budgets: Sequence[int],
    seed: int,
    initial_seed_size: int,
    model_name: str,
    device: str | None,
    adapter_kind: str = "distilbert",
) -> list[dict[str, Any]]:
    sample_by_id = {s.sample_id: s for s in dataset.samples}
    train_ids = sorted(s.sample_id for s in dataset.samples if s.split == "train")
    test_ids = sorted(s.sample_id for s in dataset.samples if s.split == "test")
    train_samples = [sample_by_id[i] for i in train_ids]
    test_texts = [sample_by_id[i].text for i in test_ids]
    test_labels = [sample_by_id[i].label for i in test_ids]

    max_budget = min(max(budgets), len(train_ids))
    usable = [b for b in sorted(set(budgets)) if initial_seed_size <= b <= max_budget]

    labeled_ids = choose_initial_seed(train_samples, dataset.labels, initial_seed_size, seed)
    provider = Provider([s for s in dataset.samples if s.split == "train"])
    model = _make_adapter(adapter_kind, dataset.labels, seed=seed, model_name=model_name, device=device)
    scheduler = StrategyScheduler(SchedulerConfig(mode="single", strategy=strategy_name))
    label_schema = LabelSchema(task="text_classification", labels=dataset.labels)

    def train_eval() -> dict[str, float]:
        model.fit(
            [sample_by_id[i].text for i in labeled_ids],
            [sample_by_id[i].label for i in labeled_ids],
        )
        return model.evaluate(test_texts, test_labels)

    rows: list[dict[str, Any]] = []
    current: dict[str, float] | None = None
    ready = False
    for budget in usable:
        started = time.perf_counter()
        if not ready:
            current = train_eval()
            ready = True
        pool_ids = [i for i in train_ids if i not in set(labeled_ids)]
        to_select = min(budget - len(labeled_ids), len(pool_ids))
        selected: list[str] = []
        snapshot: dict[str, Any] = {"no_acquisition_needed": True}
        if to_select > 0:
            context = SelectionContext(
                provider=provider,
                model=model,
                label_schema=label_schema,
                prediction_cache=None,
                embedding_cache=None,
                labeled_ids=labeled_ids,
                last_metrics=current or {},
            )
            selected, snapshot = scheduler.select_batch(pool_ids, to_select, context, state={})
            labeled_ids.extend(selected)
            ready = False
        if not ready:
            current = train_eval()
            ready = True
        assert current is not None
        rows.append(
            {
                "dataset": dataset.name,
                "strategy": strategy_name,
                "seed": seed,
                "budget": len(labeled_ids),
                "requested_budget": budget,
                "initial_seed_size": initial_seed_size,
                "accuracy": current["accuracy"],
                "macro_f1": current["macro_f1"],
                "weighted_f1": current["weighted_f1"],
                "balanced_accuracy": current["balanced_accuracy"],
                "macro_recall": current["macro_recall"],
                "selected_count": len(selected),
                "runtime_seconds": round(time.perf_counter() - started, 3),
            }
        )
    return rows


# --------------------------------------------------------------------------- presets / jobs

_PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "datasets": ["synthetic"],
        "strategies": ["random", "entropy", "badge"],
        "seeds": [13],
        "budgets": [20, 40],
        "initial_seed_size": 9,
        "max_train": None,
        "max_test": None,
    },
    "deadline": {
        "datasets": ["ag_news", "sst2"],
        "strategies": ["random", "entropy", "margin", "least_confidence", "coreset_kcenter", "badge"],
        "seeds": [13, 21, 34],
        "budgets": [50, 100, 200, 400],
        "initial_seed_size": 20,
        "max_train": 2000,
        "max_test": 500,
    },
    # The repo's cold-start-aware / composite strategies vs random and vanilla entropy, same
    # protocol as `deadline` so the two runs are directly comparable.
    "mitigation": {
        "datasets": ["ag_news", "sst2"],
        "strategies": ["random", "entropy", "adaptive_uncertainty_diversity",
                       "class_group_balanced_entropy", "density_weighted_diversity"],
        "seeds": [13, 21, 34],
        "budgets": [50, 100, 200, 400],
        "initial_seed_size": 20,
        "max_train": 2000,
        "max_test": 500,
    },
}

_METRIC_FIELDS = [
    "dataset", "strategy", "seed", "budget", "requested_budget", "initial_seed_size",
    "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "macro_recall",
    "selected_count", "runtime_seconds",
]


def _job_list(cfg: dict[str, Any]) -> list[tuple[str, str, int]]:
    jobs: list[tuple[str, str, int]] = []
    for ds in cfg["datasets"]:
        for strat in cfg["strategies"]:
            for seed in cfg["seeds"]:
                jobs.append((ds, strat, int(seed)))
    return jobs


def _append_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in _METRIC_FIELDS})


def _merge(output_dir: Path) -> None:
    merged = output_dir / "metrics.csv"
    shard_files = sorted(output_dir.glob("metrics_shard*.csv"))
    seen: set[tuple] = set()
    rows: list[dict[str, str]] = []
    for f in shard_files:
        with f.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = (row["dataset"], row["strategy"], row["seed"], row["budget"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    with merged.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in _METRIC_FIELDS})
    print(f"[merge] wrote {len(rows)} rows -> {merged}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer active-learning benchmark")
    parser.add_argument("--preset", choices=list(_PRESETS), default="smoke")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--adapter", choices=["distilbert", "fake"], default="distilbert",
                        help="'fake' = numpy nearest-centroid stub for offline harness smoke tests")
    parser.add_argument("--device", default=None, help="cuda / cpu (auto if omitted)")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--datasets", default=None, help="comma-separated override")
    parser.add_argument("--strategies", default=None, help="comma-separated override")
    parser.add_argument("--seeds", default=None, help="comma-separated override")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        _merge(output_dir)
        return

    cfg = dict(_PRESETS[args.preset])
    if args.datasets:
        cfg["datasets"] = args.datasets.split(",")
    if args.strategies:
        cfg["strategies"] = args.strategies.split(",")
    if args.seeds:
        cfg["seeds"] = [int(x) for x in args.seeds.split(",")]

    suffix = "" if args.shard_count <= 1 else f"_shard{args.shard_index}"
    metrics_path = output_dir / f"metrics{suffix}.csv"
    checkpoint_path = output_dir / f"checkpoint{suffix}.json"

    done: set[str] = set()
    if checkpoint_path.exists():
        done = set(json.loads(checkpoint_path.read_text(encoding="utf-8")).get("done", []))

    jobs = _job_list(cfg)
    my_jobs = [job for i, job in enumerate(jobs) if i % args.shard_count == args.shard_index]
    print(f"[shard {args.shard_index}/{args.shard_count}] {len(my_jobs)} jobs ({len(done)} already done)")

    dataset_cache: dict[str, Dataset] = {}
    overall_start = time.perf_counter()
    for ds_name, strat, seed in my_jobs:
        job_key = f"{ds_name}|{strat}|{seed}"
        if job_key in done:
            continue
        if ds_name not in dataset_cache:
            print(f"[load] {ds_name}")
            dataset_cache[ds_name] = load_dataset_by_name(ds_name, cfg["max_train"], cfg["max_test"])
        dataset = dataset_cache[ds_name]
        t0 = time.perf_counter()
        rows = run_one_curve(
            dataset, strat, cfg["budgets"], seed, cfg["initial_seed_size"],
            args.model_name, args.device, adapter_kind=args.adapter,
        )
        _append_rows(metrics_path, rows)
        done.add(job_key)
        checkpoint_path.write_text(json.dumps({"done": sorted(done)}, indent=2), encoding="utf-8")
        elapsed = time.perf_counter() - t0
        final = rows[-1] if rows else {}
        print(
            f"[done] {job_key} in {elapsed:.1f}s "
            f"(final macro_f1={final.get('macro_f1', float('nan')):.4f})"
        )

    print(f"[shard {args.shard_index}] complete in {time.perf_counter() - overall_start:.1f}s")
    if args.shard_count <= 1:
        # single-process convenience: also emit a canonical metrics.csv copy
        if metrics_path.name != "metrics.csv":
            _merge(output_dir)


if __name__ == "__main__":
    main()
