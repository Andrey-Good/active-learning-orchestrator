from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_BUDGETS = [16, 32, 48, 64, 96]
SMOKE_BUDGETS = [12, 24, 36]
REAL_DATASET_BUDGETS = [100, 200, 300, 500, 800]
BENCHMARK_ARTIFACT_SCHEMA_VERSION = 1
REAL_STANDARD_PRESETS = {"real_medium", "real_full"}
REAL_STANDARD_MIN_SEEDS = 3
CALIBRATION_COLUMNS = ("multiclass_brier_score", "nll", "ece")
CLAIM_CATEGORY_ACTIVE_LEARNING_QUALITY = "active_learning_quality"
CLAIM_CATEGORY_END_TO_END_PUBLIC_PROJECT_WORKFLOW = "end_to_end_public_project_workflow"

TfidfVectorizer: Any = None
LogisticRegression: Any = None
Pipeline: Any = None
accuracy_score: Any = None
balanced_accuracy_score: Any = None
f1_score: Any = None
recall_score: Any = None
ActiveLearningProject: Any = None
AnnotationPolicy: Any = None
CacheConfig: Any = None
FingerprintConfig: Any = None
LabelBackendConfig: Any = None
LabelSchema: Any = None
SchedulerConfig: Any = None
SelectionContext: Any = None
SplitConfig: Any = None
StopCriteria: Any = None
StrategyScheduler: Any = None
AnnotationRecord: Any = None
DataSample: Any = None
RoundProgress: Any = None
RoundPullResult: Any = None
RoundPushResult: Any = None
SklearnTextClassifierAdapter: Any = None


def ensure_benchmark_dependencies() -> None:
    global TfidfVectorizer
    global LogisticRegression
    global Pipeline
    global accuracy_score
    global balanced_accuracy_score
    global f1_score
    global recall_score
    global ActiveLearningProject
    global AnnotationPolicy
    global CacheConfig
    global FingerprintConfig
    global LabelBackendConfig
    global LabelSchema
    global SchedulerConfig
    global SelectionContext
    global SplitConfig
    global StopCriteria
    global StrategyScheduler
    global AnnotationRecord
    global DataSample
    global RoundProgress
    global RoundPullResult
    global RoundPushResult
    global SklearnTextClassifierAdapter

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
        from sklearn.linear_model import LogisticRegression as _LogisticRegression
        from sklearn.metrics import accuracy_score as _accuracy_score
        from sklearn.metrics import balanced_accuracy_score as _balanced_accuracy_score
        from sklearn.metrics import f1_score as _f1_score
        from sklearn.metrics import recall_score as _recall_score
        from sklearn.pipeline import Pipeline as _Pipeline
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Missing benchmark dependency. Run with the project environment, for example: "
            "`uv run python benchmarks/sdk_first_benchmark.py --preset smoke`."
        ) from error

    from active_learning_sdk.backends.base import RoundProgress as _RoundProgress
    from active_learning_sdk.backends.base import RoundPullResult as _RoundPullResult
    from active_learning_sdk.backends.base import RoundPushResult as _RoundPushResult
    from active_learning_sdk.adapters import SklearnTextClassifierAdapter as _SklearnTextClassifierAdapter
    from active_learning_sdk.configs import AnnotationPolicy as _AnnotationPolicy
    from active_learning_sdk.configs import CacheConfig as _CacheConfig
    from active_learning_sdk.configs import FingerprintConfig as _FingerprintConfig
    from active_learning_sdk.configs import LabelBackendConfig as _LabelBackendConfig
    from active_learning_sdk.configs import LabelSchema as _LabelSchema
    from active_learning_sdk.configs import SchedulerConfig as _SchedulerConfig
    from active_learning_sdk.configs import SplitConfig as _SplitConfig
    from active_learning_sdk.configs import StopCriteria as _StopCriteria
    from active_learning_sdk.engine import SelectionContext as _SelectionContext
    from active_learning_sdk.engine import StrategyScheduler as _StrategyScheduler
    from active_learning_sdk.project import ActiveLearningProject as _ActiveLearningProject
    from active_learning_sdk.types import AnnotationRecord as _AnnotationRecord
    from active_learning_sdk.types import DataSample as _DataSample

    TfidfVectorizer = _TfidfVectorizer
    LogisticRegression = _LogisticRegression
    Pipeline = _Pipeline
    accuracy_score = _accuracy_score
    balanced_accuracy_score = _balanced_accuracy_score
    f1_score = _f1_score
    recall_score = _recall_score
    ActiveLearningProject = _ActiveLearningProject
    AnnotationPolicy = _AnnotationPolicy
    CacheConfig = _CacheConfig
    FingerprintConfig = _FingerprintConfig
    LabelBackendConfig = _LabelBackendConfig
    LabelSchema = _LabelSchema
    SchedulerConfig = _SchedulerConfig
    SelectionContext = _SelectionContext
    SplitConfig = _SplitConfig
    StopCriteria = _StopCriteria
    StrategyScheduler = _StrategyScheduler
    AnnotationRecord = _AnnotationRecord
    DataSample = _DataSample
    RoundProgress = _RoundProgress
    RoundPullResult = _RoundPullResult
    RoundPushResult = _RoundPushResult
    SklearnTextClassifierAdapter = _SklearnTextClassifierAdapter


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    text: str
    label: str
    group_id: str
    split: str


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    description: str
    labels: list[str]
    samples: list[BenchmarkSample]
    rare_label: str | None = None


@dataclass(frozen=True)
class SyntheticRecord:
    text: str
    label: str
    split: str
    internal_group_id: str


@dataclass(frozen=True)
class RealDatasetSpec:
    name: str
    hf_path: str
    hf_config: str
    splits: tuple[str, ...]
    text_column: str
    label_column: str
    label_name_column: str | None
    description: str
    sanity_easy_coverage: bool = False


@dataclass(frozen=True)
class StrategySpec:
    name: str
    scheduler_config: SchedulerConfig


@dataclass(frozen=True)
class StopPolicySpec:
    name: str
    policy_type: str
    metric_name: str = "macro_f1"
    min_budget: int = 0
    patience: int = 1
    min_delta: float = 0.0


DATASET_ID_PREFIXES = {
    "separable_topics": "ds001",
    "rare_class_trap": "ds002",
    "grouped_duplicates": "ds003",
    "clinc_oos_imbalanced": "ds101",
    "clinc_oos_plus": "ds102",
    "banking77": "ds103",
    "dair_ai_emotion": "ds104",
}


REAL_DATASET_REGISTRY = {
    "clinc_oos_imbalanced": RealDatasetSpec(
        name="clinc_oos_imbalanced",
        hf_path="clinc/clinc_oos",
        hf_config="imbalanced",
        splits=("train", "validation", "test"),
        text_column="text",
        label_column="intent",
        label_name_column=None,
        description="CLINC OOS imbalanced intent classification dataset.",
    ),
    "clinc_oos_plus": RealDatasetSpec(
        name="clinc_oos_plus",
        hf_path="clinc/clinc_oos",
        hf_config="plus",
        splits=("train", "validation", "test"),
        text_column="text",
        label_column="intent",
        label_name_column=None,
        description="CLINC OOS plus intent classification dataset.",
    ),
    "banking77": RealDatasetSpec(
        name="banking77",
        hf_path="mteb/banking77",
        hf_config="default",
        splits=("train", "test"),
        text_column="text",
        label_column="label",
        label_name_column="label_text",
        description="Banking77 customer-service intent classification dataset.",
    ),
    "dair_ai_emotion": RealDatasetSpec(
        name="dair_ai_emotion",
        hf_path="dair-ai/emotion",
        hf_config="split",
        splits=("train", "validation", "test"),
        text_column="text",
        label_column="label",
        label_name_column=None,
        description="DAIR AI emotion classification dataset, useful as an easier real-data sanity check.",
        sanity_easy_coverage=True,
    ),
}


class InMemoryBenchmarkProvider:
    def __init__(self, samples: Sequence[BenchmarkSample]) -> None:
        self._samples = {sample.sample_id: sample for sample in samples}

    def iter_sample_ids(self) -> Iterable[str]:
        return iter(self._samples.keys())

    def get_sample(self, sample_id: str) -> DataSample:
        sample = self._samples[str(sample_id)]
        meta = {"split": sample.split}
        return DataSample(
            sample_id=sample.sample_id,
            data={"text": sample.text},
            meta=meta,
            group_id=sample.group_id,
        )

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        return [self.get_sample(sample_id) for sample_id in sample_ids]

    def schema(self) -> dict[str, str]:
        return {"sample_id": "str", "text": "str", "split": "str", "group_id": "str"}


class SklearnTextBenchmarkAdapter:
    COMMITTEE_MEMBER_COUNT = 5
    BADGE_GRADIENT_EMBEDDING_DIM = 1024

    def __init__(self, labels: Sequence[str], seed: int) -> None:
        self.labels = list(labels)
        self.seed = seed
        self.fit_count = 0
        self.pipeline: Pipeline | None = None

    def fit(self, texts: Sequence[str], labels: Sequence[str], **_: Any) -> None:
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                (
                    "clf",
                    LogisticRegression(
                        C=2.0,
                        max_iter=500,
                        random_state=self.seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        self.pipeline.fit(list(texts), list(labels))
        self.fit_count += 1

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        if self.pipeline is None:
            raise RuntimeError("Model must be fitted before predict_proba is called.")
        raw = self.pipeline.predict_proba(list(texts))
        model_classes = [str(label) for label in self.pipeline.named_steps["clf"].classes_]
        class_index = {label: idx for idx, label in enumerate(model_classes)}

        aligned: list[list[float]] = []
        for row in raw:
            output_row = []
            for label in self.labels:
                idx = class_index.get(label)
                output_row.append(float(row[idx]) if idx is not None else 0.0)
            row_sum = sum(output_row)
            if row_sum <= 0:
                output_row = [1.0 / len(self.labels)] * len(self.labels)
            elif not math.isclose(row_sum, 1.0):
                output_row = [value / row_sum for value in output_row]
            aligned.append(output_row)
        return aligned

    def predict_stochastic(
        self,
        texts: Sequence[str],
        n: int = 10,
        batch_size: int = 32,
    ) -> list[list[list[float]]]:
        # Benchmark proxy only: deterministic perturbations around logistic-regression
        # probabilities stand in for MC-dropout passes; this is not real MC-dropout.
        return self._prediction_proxy_cube(texts, member_count=n, batch_size=batch_size, salt="stochastic")

    def predict_committee(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> list[list[list[float]]]:
        # Benchmark proxy only: deterministic perturbations around one fitted model
        # stand in for committee members; this is not an independently trained committee.
        return self._prediction_proxy_cube(
            texts,
            member_count=self.COMMITTEE_MEMBER_COUNT,
            batch_size=batch_size,
            salt="committee",
        )

    def _prediction_proxy_cube(
        self,
        texts: Sequence[str],
        *,
        member_count: int,
        batch_size: int,
        salt: str,
    ) -> list[list[list[float]]]:
        if member_count < 1:
            raise RuntimeError("Prediction proxy member_count must be positive.")

        text_list = list(texts)
        base_rows = self.predict_proba(text_list, batch_size=batch_size)
        cube: list[list[list[float]]] = []
        for text_index, (text, base_row) in enumerate(zip(text_list, base_rows)):
            sample_rows: list[list[float]] = []
            for member_index in range(member_count):
                rng = random.Random(_stable_seed(self.seed, salt, str(self.fit_count), str(text_index), str(member_index), text))
                sample_rows.append(self._perturb_probability_row(base_row, rng))
            cube.append(sample_rows)
        return cube

    def _perturb_probability_row(self, row: Sequence[float], rng: random.Random) -> list[float]:
        adjusted: list[float] = []
        for value in row:
            multiplier = 0.75 + (0.5 * rng.random())
            offset = 0.01 * rng.random()
            adjusted.append(max(0.0, (float(value) * multiplier) + offset))
        return self._normalize_strict_probability_row(adjusted)

    def _normalize_strict_probability_row(self, row: Sequence[float]) -> list[float]:
        cleaned = [float(value) for value in row]
        if any((not math.isfinite(value)) or value < 0.0 for value in cleaned):
            raise RuntimeError("Prediction proxy probabilities must be finite and non-negative.")

        row_sum = sum(cleaned)
        if row_sum <= 0.0:
            cleaned = [1.0 / len(self.labels)] * len(self.labels)
        else:
            cleaned = [value / row_sum for value in cleaned]

        drift = 1.0 - sum(cleaned)
        if cleaned:
            cleaned[-1] += drift
        if any((not math.isfinite(value)) or value < 0.0 for value in cleaned):
            raise RuntimeError("Prediction proxy normalization produced invalid probabilities.")
        return cleaned

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        if self.pipeline is None:
            raise RuntimeError("Model must be fitted before embed is called.")
        vectorizer = self.pipeline.named_steps.get("tfidf")
        if vectorizer is None:
            raise RuntimeError("Model pipeline must contain a fitted 'tfidf' step before embed is called.")
        matrix = vectorizer.transform(list(texts))
        dense_matrix = matrix.toarray() if hasattr(matrix, "toarray") else matrix
        return [[float(value) for value in row] for row in dense_matrix]

    def gradient_embed(
        self,
        texts: Sequence[str],
        labels: Sequence[str] | None = None,
        batch_size: int = 32,
    ) -> list[list[float]]:
        del batch_size
        if self.pipeline is None:
            raise RuntimeError("Model must be fitted before gradient_embed is called.")
        vectorizer = self.pipeline.named_steps.get("tfidf")
        if vectorizer is None:
            raise RuntimeError("Model pipeline must contain a fitted 'tfidf' step before gradient_embed is called.")

        text_list = list(texts)
        probabilities = self.predict_proba(text_list)
        feature_matrix = vectorizer.transform(text_list)
        if hasattr(feature_matrix, "tocsr"):
            feature_matrix = feature_matrix.tocsr()
        label_index = {label: index for index, label in enumerate(self.labels)}

        embeddings: list[list[float]] = []
        for row_index, probability_row in enumerate(probabilities):
            pseudo_label_index = max(range(len(probability_row)), key=lambda index: probability_row[index])
            if labels is not None and row_index < len(labels):
                pseudo_label_index = label_index.get(str(labels[row_index]), pseudo_label_index)

            # Benchmark proxy only: this approximates BADGE's classifier-head gradient by
            # hashing sparse TF-IDF classifier-head gradients into a bounded vector.
            # It is not neural autograd and should not be used as a production gradient
            # embedding, but it preserves BADGE's residual-times-feature geometry without
            # materializing num_labels * vocabulary_size dense vectors.
            embedding = [0.0 for _ in range(self.BADGE_GRADIENT_EMBEDDING_DIM)]
            row = feature_matrix.getrow(row_index) if hasattr(feature_matrix, "getrow") else feature_matrix[row_index]
            feature_indices = list(getattr(row, "indices", []))
            feature_values = list(getattr(row, "data", []))
            if not feature_indices:
                dense_row = row.toarray()[0] if hasattr(row, "toarray") else row
                feature_indices = [index for index, value in enumerate(dense_row) if float(value) != 0.0]
                feature_values = [float(dense_row[index]) for index in feature_indices]
            for class_index, probability in enumerate(probability_row):
                residual = float(probability) - (1.0 if class_index == pseudo_label_index else 0.0)
                if residual == 0.0:
                    continue
                for feature_index, feature_value in zip(feature_indices, feature_values):
                    bucket_seed = _stable_seed(
                        self.seed,
                        "badge-gradient-bucket",
                        str(class_index),
                        str(feature_index),
                    )
                    sign_seed = _stable_seed(
                        self.seed,
                        "badge-gradient-sign",
                        str(class_index),
                        str(feature_index),
                    )
                    bucket = bucket_seed % self.BADGE_GRADIENT_EMBEDDING_DIM
                    sign = 1.0 if sign_seed % 2 == 0 else -1.0
                    embedding[bucket] += sign * float(feature_value) * residual
            embeddings.append(embedding)
        return embeddings

    def evaluate(self, texts: Sequence[str], labels: Sequence[str]) -> dict[str, float]:
        if self.pipeline is None:
            raise RuntimeError("Model must be fitted before evaluate is called.")
        predictions = [str(value) for value in self.pipeline.predict(list(texts))]
        y_true = [str(value) for value in labels]
        return {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
            "macro_f1": float(f1_score(y_true, predictions, labels=self.labels, average="macro", zero_division=0)),
            "weighted_f1": float(
                f1_score(y_true, predictions, labels=self.labels, average="weighted", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(y_true, predictions, labels=self.labels, average="macro", zero_division=0)
            ),
        }

    def get_model_id(self) -> str:
        return f"sklearn-tfidf-logreg-seed-{self.seed}-fit-{self.fit_count}"

    def get_embedding_config(self) -> str:
        return "tfidf-ngram-1-2-min-df-1"


class OracleProjectSmokeBackend:
    """Benchmark-only backend that labels pushed samples from a private oracle map."""

    def __init__(self, label_by_id: dict[str, str]) -> None:
        self._label_by_id = dict(label_by_id)
        self._ready = False
        self._tasks_by_round: dict[str, dict[str, str]] = {}
        self._annotations_by_round: dict[str, dict[str, list[Any]]] = {}
        self.pushed_sample_ids: list[str] = []
        self.pulled_sample_ids: list[str] = []

    def ensure_ready(self, label_schema: LabelSchema) -> dict[str, Any]:
        label_schema.validate()
        self._ready = True
        return {"backend": "benchmark_oracle", "auto_labels_after_push": True}

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: dict[str, Any] | None = None,
    ) -> RoundPushResult:
        del prelabels
        self._require_ready()
        task_ids: dict[str, str] = {}
        round_tasks = self._tasks_by_round.setdefault(round_id, {})
        round_annotations = self._annotations_by_round.setdefault(round_id, {})
        for index, sample in enumerate(samples):
            task_id = round_tasks.setdefault(sample.sample_id, f"oracle:{round_id}:{sample.sample_id}")
            task_ids[sample.sample_id] = task_id
            if sample.sample_id not in round_annotations:
                label = self._label_by_id[sample.sample_id]
                round_annotations[sample.sample_id] = [
                    AnnotationRecord(
                        annotator_id="benchmark_oracle",
                        created_at=1_800_000_000.0 + index,
                        value=label,
                        score=1.0,
                    )
                ]
                self.pushed_sample_ids.append(sample.sample_id)
        return RoundPushResult(
            task_ids=task_ids,
            backend_round_ref={"backend": "benchmark_oracle", "round_id": round_id},
        )

    def poll_round(self, round_id: str, task_ids: dict[str, str], policy: AnnotationPolicy) -> RoundProgress:
        self._require_ready()
        round_annotations = self._annotations_by_round.get(round_id, {})
        ready_sample_ids = [
            sample_id
            for sample_id in task_ids
            if len(round_annotations.get(sample_id, [])) >= policy.min_votes
        ]
        return RoundProgress(
            total=len(task_ids),
            done=len(ready_sample_ids),
            ready_sample_ids=ready_sample_ids,
            details={"backend": "benchmark_oracle", "round_id": round_id},
        )

    def pull_round(self, round_id: str, task_ids: dict[str, str]) -> RoundPullResult:
        self._require_ready()
        round_annotations = self._annotations_by_round.get(round_id, {})
        annotations = {sample_id: list(round_annotations.get(sample_id, [])) for sample_id in task_ids}
        self.pulled_sample_ids.extend(annotations.keys())
        return RoundPullResult(
            annotations=annotations,
            backend_payload={"backend": "benchmark_oracle", "round_id": round_id, "task_count": len(task_ids)},
        )

    def close(self) -> None:
        self._ready = False

    def _require_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("OracleProjectSmokeBackend is not ready. Call ensure_ready() first.")


def _make_text(label_terms: Sequence[str], shared_terms: Sequence[str], noise_terms: Sequence[str], rng: random.Random) -> str:
    terms = [rng.choice(label_terms) for _ in range(5)]
    terms.extend(rng.choice(shared_terms) for _ in range(3))
    terms.extend(rng.choice(noise_terms) for _ in range(2))
    rng.shuffle(terms)
    return " ".join(terms)


def _opaque_sample_id(dataset_name: str, split: str, index: int) -> str:
    return f"{DATASET_ID_PREFIXES[dataset_name]}_{split}_s{index:05d}"


def _opaque_group_id(dataset_name: str, split: str, index: int) -> str:
    return f"{DATASET_ID_PREFIXES[dataset_name]}_{split}_g{index:05d}"


def _stable_seed(seed: int, *parts: str) -> int:
    value = seed
    for part in parts:
        for char in part:
            value = (value * 131 + ord(char)) % (2**32)
    return value


def _assign_opaque_ids(dataset_name: str, records: Sequence[SyntheticRecord], seed: int) -> list[BenchmarkSample]:
    samples: list[BenchmarkSample] = []
    split_order = ["train", "test"]
    remaining_splits = sorted({record.split for record in records} - set(split_order))

    for split in [*split_order, *remaining_splits]:
        split_records = [record for record in records if record.split == split]
        if not split_records:
            continue

        shuffled_records = list(split_records)
        random.Random(_stable_seed(seed, dataset_name, split, "sample-order")).shuffle(shuffled_records)

        group_ids: dict[str, str] = {}
        for record in shuffled_records:
            if record.internal_group_id not in group_ids:
                group_ids[record.internal_group_id] = _opaque_group_id(dataset_name, split, len(group_ids))

        for sample_index, record in enumerate(shuffled_records):
            samples.append(
                BenchmarkSample(
                    sample_id=_opaque_sample_id(dataset_name, split, sample_index),
                    text=record.text,
                    label=record.label,
                    group_id=group_ids[record.internal_group_id],
                    split=split,
                )
            )

    return samples


def load_hf_dataset(path: str, config: str) -> Any:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Missing optional benchmark dependency 'datasets'. Install it to run real Hugging Face datasets, "
            "or use the synthetic smoke/full presets."
        ) from error

    return load_dataset(path, config)


def _label_to_private_name(raw_label: Any, label_feature: Any, explicit_label_name: Any = None) -> str:
    if explicit_label_name is not None:
        return str(explicit_label_name)
    if label_feature is not None and hasattr(label_feature, "int2str"):
        try:
            return str(label_feature.int2str(int(raw_label)))
        except (TypeError, ValueError):
            pass
    return str(raw_label)


def _label_feature_for_split(split_dataset: Any, label_column: str) -> Any:
    features = getattr(split_dataset, "features", None)
    if features is None:
        return None
    return features.get(label_column)


def make_real_dataset(
    dataset_name: str,
    seed: int,
    *,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> BenchmarkDataset:
    spec = REAL_DATASET_REGISTRY[dataset_name]
    dataset_dict = load_hf_dataset(spec.hf_path, spec.hf_config)
    records: list[SyntheticRecord] = []

    for split in spec.splits:
        if split not in dataset_dict:
            raise RuntimeError(f"{dataset_name} is missing expected Hugging Face split: {split}")
        split_dataset = dataset_dict[split]
        label_feature = _label_feature_for_split(split_dataset, spec.label_column)
        split_records: list[SyntheticRecord] = []
        for index, row in enumerate(split_dataset):
            if spec.text_column not in row or spec.label_column not in row:
                raise RuntimeError(
                    f"{dataset_name}/{split} row {index} is missing "
                    f"{spec.text_column!r} or {spec.label_column!r}."
            )
            explicit_label_name = row.get(spec.label_name_column) if spec.label_name_column else None
            label = _label_to_private_name(row[spec.label_column], label_feature, explicit_label_name)
            split_records.append(
                SyntheticRecord(
                    text=str(row[spec.text_column]),
                    label=label,
                    split=split,
                    internal_group_id=f"{split}:row:{index}",
                )
            )

        random.Random(_stable_seed(seed, dataset_name, split, "real-record-order")).shuffle(split_records)
        if split == "train" and max_train_samples is not None:
            split_records = split_records[:max_train_samples]
        if split == "test" and max_test_samples is not None:
            split_records = split_records[:max_test_samples]
        records.extend(split_records)

    return BenchmarkDataset(
        name=dataset_name,
        description=spec.description,
        labels=sorted({record.label for record in records}),
        samples=_assign_opaque_ids(dataset_name, records, seed),
    )


def make_clinc_oos_imbalanced(seed: int) -> BenchmarkDataset:
    return make_real_dataset("clinc_oos_imbalanced", seed)


def make_clinc_oos_plus(seed: int) -> BenchmarkDataset:
    return make_real_dataset("clinc_oos_plus", seed)


def make_banking77(seed: int) -> BenchmarkDataset:
    return make_real_dataset("banking77", seed)


def make_dair_ai_emotion(seed: int) -> BenchmarkDataset:
    return make_real_dataset("dair_ai_emotion", seed)


def build_benchmark_dataset(
    dataset_name: str,
    seed: int,
    *,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> BenchmarkDataset:
    if dataset_name in REAL_DATASET_REGISTRY:
        return make_real_dataset(
            dataset_name,
            seed,
            max_train_samples=max_train_samples,
            max_test_samples=max_test_samples,
        )
    return DATASET_BUILDERS[dataset_name](seed)


def make_separable_topics(seed: int) -> BenchmarkDataset:
    rng = random.Random(seed + 101)
    dataset_name = "separable_topics"
    labels = ["sports", "science", "finance"]
    terms = {
        "sports": ["goal", "match", "coach", "league", "score", "team"],
        "science": ["orbit", "molecule", "lab", "theory", "sensor", "quantum"],
        "finance": ["market", "equity", "bond", "profit", "trade", "bank"],
    }
    shared = ["update", "report", "analysis", "today", "brief"]
    noise = ["north", "green", "rapid", "public", "weekly"]
    records: list[SyntheticRecord] = []
    for split, per_label in [("train", 72), ("test", 36)]:
        for label in labels:
            for idx in range(per_label):
                text = _make_text(terms[label], shared, noise, rng)
                internal_group_id = f"{split}:{label}:group:{idx // 6}"
                records.append(SyntheticRecord(text, label, split, internal_group_id))
    return BenchmarkDataset(
        name=dataset_name,
        description="Balanced three-class text data with strong label-specific vocabulary.",
        labels=labels,
        samples=_assign_opaque_ids(dataset_name, records, seed),
    )


def make_rare_class(seed: int) -> BenchmarkDataset:
    rng = random.Random(seed + 202)
    dataset_name = "rare_class_trap"
    labels = ["routine", "urgent", "rare"]
    terms = {
        "routine": ["ticket", "queue", "normal", "standard", "update", "request"],
        "urgent": ["incident", "failure", "urgent", "blocked", "outage", "escalate"],
        "rare": ["fraud", "breach", "anomaly", "rare", "forensic", "suspicious"],
    }
    shared = ["customer", "case", "service", "message", "review"]
    noise = ["blue", "east", "short", "manual", "daily"]
    counts = {
        "train": {"routine": 130, "urgent": 70, "rare": 20},
        "test": {"routine": 65, "urgent": 35, "rare": 20},
    }
    records: list[SyntheticRecord] = []
    for split, split_counts in counts.items():
        for label, count in split_counts.items():
            for idx in range(count):
                # Rare samples deliberately share some urgent vocabulary to test discovery quality.
                label_terms = terms[label]
                if label == "rare" and idx % 3 == 0:
                    label_terms = [*terms["rare"], "incident", "escalate"]
                text = _make_text(label_terms, shared, noise, rng)
                internal_group_id = f"{split}:{label}:group:{idx // 5}"
                records.append(SyntheticRecord(text, label, split, internal_group_id))
    return BenchmarkDataset(
        name=dataset_name,
        description="Imbalanced data with a rare class that partially overlaps an urgent majority class.",
        labels=labels,
        samples=_assign_opaque_ids(dataset_name, records, seed),
        rare_label="rare",
    )


def make_grouped_duplicates(seed: int) -> BenchmarkDataset:
    rng = random.Random(seed + 303)
    dataset_name = "grouped_duplicates"
    labels = ["alpha", "beta", "gamma"]
    terms = {
        "alpha": ["alpha", "apple", "amber", "axis", "archive"],
        "beta": ["beta", "bridge", "binary", "buffer", "branch"],
        "gamma": ["gamma", "graph", "garden", "globe", "grid"],
    }
    shared = ["memo", "cluster", "near", "copy", "note"]
    noise = ["low", "wide", "fresh", "silent", "urban"]
    records: list[SyntheticRecord] = []
    for split, groups_per_label, variants_per_group in [("train", 18, 4), ("test", 9, 3)]:
        for label in labels:
            for group_idx in range(groups_per_label):
                base = _make_text(terms[label], shared, noise, rng)
                internal_group_id = f"{split}:{label}:group:{group_idx}"
                for variant_idx in range(variants_per_group):
                    text = f"{base} variant_{variant_idx % 2} {rng.choice(shared)}"
                    records.append(SyntheticRecord(text, label, split, internal_group_id))
    return BenchmarkDataset(
        name=dataset_name,
        description="Balanced labels with near-duplicate groups for concentration diagnostics.",
        labels=labels,
        samples=_assign_opaque_ids(dataset_name, records, seed),
    )


DATASET_BUILDERS = {
    "separable_topics": make_separable_topics,
    "rare_class_trap": make_rare_class,
    "grouped_duplicates": make_grouped_duplicates,
    "clinc_oos_imbalanced": make_clinc_oos_imbalanced,
    "clinc_oos_plus": make_clinc_oos_plus,
    "banking77": make_banking77,
    "dair_ai_emotion": make_dair_ai_emotion,
}

SYNTHETIC_DATASET_NAMES = ("separable_topics", "rare_class_trap", "grouped_duplicates")
REAL_DATASET_NAMES = tuple(REAL_DATASET_REGISTRY)


def _matching_labels(value: Any, labels: Sequence[str]) -> list[str]:
    normalized = str(value).lower()
    return sorted(label for label in labels if label.lower() in normalized)


def _compressed_label_order(labels: Sequence[str]) -> list[str]:
    compressed: list[str] = []
    for label in labels:
        if not compressed or compressed[-1] != label:
            compressed.append(label)
    return compressed


def _validate_mixed_label_order(
    dataset: BenchmarkDataset,
    split: str,
    kind: str,
    ordered_labels: Sequence[str],
) -> dict[str, Any]:
    unique_labels = sorted(set(ordered_labels))
    compressed = _compressed_label_order(ordered_labels)
    if len(unique_labels) > 1 and len(ordered_labels) > len(unique_labels) and len(compressed) <= len(unique_labels):
        raise RuntimeError(
            f"{dataset.name}/{split} {kind} order still forms label blocks: "
            f"{' -> '.join(compressed)}"
        )
    return {
        "kind": kind,
        "split": split,
        "item_count": len(ordered_labels),
        "unique_label_count": len(unique_labels),
        "label_transitions": max(0, len(compressed) - 1),
        "compressed_prefix": compressed[:20],
    }


def validate_acquisition_surface(dataset: BenchmarkDataset) -> dict[str, Any]:
    provider = InMemoryBenchmarkProvider(dataset.samples)
    schema = provider.schema()
    label_leaks: list[dict[str, Any]] = []

    schema_items = [*schema.keys(), *schema.values()]
    for item in schema_items:
        matches = _matching_labels(item, dataset.labels)
        if matches:
            label_leaks.append({"surface": "schema", "value": item, "labels": matches})

    for sample in dataset.samples:
        data_sample = provider.get_sample(sample.sample_id)
        visible_values = {
            "sample_id": data_sample.sample_id,
            "group_id": data_sample.group_id,
            **{f"meta.{key}": key for key in data_sample.meta},
            **{f"meta_value.{key}": value for key, value in data_sample.meta.items()},
        }
        for surface, value in visible_values.items():
            matches = _matching_labels(value, dataset.labels)
            if matches:
                label_leaks.append(
                    {
                        "sample_id": sample.sample_id,
                        "surface": surface,
                        "value": value,
                        "labels": matches,
                    }
                )

    if label_leaks:
        raise RuntimeError(f"{dataset.name} exposes labels in acquisition-visible fields: {label_leaks[:5]}")

    order_checks: list[dict[str, Any]] = []
    for split in sorted({sample.split for sample in dataset.samples}):
        split_samples = [sample for sample in dataset.samples if sample.split == split]
        samples_by_id = sorted(split_samples, key=lambda sample: sample.sample_id)
        order_checks.append(
            _validate_mixed_label_order(
                dataset,
                split,
                "sample_id",
                [sample.label for sample in samples_by_id],
            )
        )

        group_labels: dict[str, set[str]] = defaultdict(set)
        for sample in split_samples:
            group_labels[sample.group_id].add(sample.label)
        labels_by_sorted_group = [
            "|".join(sorted(group_labels[group_id]))
            for group_id in sorted(group_labels)
        ]
        order_checks.append(
            _validate_mixed_label_order(
                dataset,
                split,
                "group_id",
                labels_by_sorted_group,
            )
        )

    return {
        "dataset": dataset.name,
        "samples": len(dataset.samples),
        "labels": dataset.labels,
        "checks": order_checks,
        "label_leak_count": 0,
    }


def strategy_specs() -> dict[str, StrategySpec]:
    return {
        "random": StrategySpec("random", SchedulerConfig(mode="single", strategy="random")),
        "adaptive_uncertainty_diversity": StrategySpec(
            "adaptive_uncertainty_diversity",
            SchedulerConfig(mode="single", strategy="adaptive_uncertainty_diversity"),
        ),
        "entropy": StrategySpec("entropy", SchedulerConfig(mode="single", strategy="entropy")),
        "group_diverse_entropy": StrategySpec(
            "group_diverse_entropy", SchedulerConfig(mode="single", strategy="group_diverse_entropy")
        ),
        "class_balanced_entropy": StrategySpec(
            "class_balanced_entropy", SchedulerConfig(mode="single", strategy="class_balanced_entropy")
        ),
        "class_group_balanced_entropy": StrategySpec(
            "class_group_balanced_entropy", SchedulerConfig(mode="single", strategy="class_group_balanced_entropy")
        ),
        "margin": StrategySpec("margin", SchedulerConfig(mode="single", strategy="margin")),
        "least_confidence": StrategySpec(
            "least_confidence", SchedulerConfig(mode="single", strategy="least_confidence")
        ),
        "coreset_kcenter": StrategySpec(
            "coreset_kcenter", SchedulerConfig(mode="single", strategy="coreset_kcenter")
        ),
        "embedding_kmeans_pp": StrategySpec(
            "embedding_kmeans_pp", SchedulerConfig(mode="single", strategy="embedding_kmeans_pp")
        ),
        "max_min_embedding": StrategySpec(
            "max_min_embedding", SchedulerConfig(mode="single", strategy="max_min_embedding")
        ),
        "deduplicate_near_neighbors": StrategySpec(
            "deduplicate_near_neighbors", SchedulerConfig(mode="single", strategy="deduplicate_near_neighbors")
        ),
        "density_weighted_diversity": StrategySpec(
            "density_weighted_diversity", SchedulerConfig(mode="single", strategy="density_weighted_diversity")
        ),
        "badge": StrategySpec("badge", SchedulerConfig(mode="single", strategy="badge")),
        "mc_dropout_entropy": StrategySpec(
            "mc_dropout_entropy", SchedulerConfig(mode="single", strategy="mc_dropout_entropy")
        ),
        "bald": StrategySpec("bald", SchedulerConfig(mode="single", strategy="bald")),
        "variation_ratio": StrategySpec(
            "variation_ratio", SchedulerConfig(mode="single", strategy="variation_ratio")
        ),
        "prediction_variance": StrategySpec(
            "prediction_variance", SchedulerConfig(mode="single", strategy="prediction_variance")
        ),
        "committee_vote_entropy": StrategySpec(
            "committee_vote_entropy", SchedulerConfig(mode="single", strategy="committee_vote_entropy")
        ),
        "committee_kl_divergence": StrategySpec(
            "committee_kl_divergence", SchedulerConfig(mode="single", strategy="committee_kl_divergence")
        ),
        "committee_pairwise_disagreement": StrategySpec(
            "committee_pairwise_disagreement",
            SchedulerConfig(mode="single", strategy="committee_pairwise_disagreement"),
        ),
        "committee_margin": StrategySpec(
            "committee_margin", SchedulerConfig(mode="single", strategy="committee_margin")
        ),
        "mix_entropy_random": StrategySpec(
            "mix_entropy_random", SchedulerConfig(mode="mix", mix={"entropy": 0.7, "random": 0.3})
        ),
        "mix_uncertainty_random": StrategySpec(
            "mix_uncertainty_random",
            SchedulerConfig(mode="mix", mix={"entropy": 0.4, "margin": 0.3, "random": 0.3}),
        ),
        "mix_group_diverse_random": StrategySpec(
            "mix_group_diverse_random",
            SchedulerConfig(mode="mix", mix={"group_diverse_entropy": 0.4, "margin": 0.3, "random": 0.3}),
        ),
        "mix_class_group_random": StrategySpec(
            "mix_class_group_random",
            SchedulerConfig(mode="mix", mix={"class_group_balanced_entropy": 0.7, "random": 0.3}),
        ),
        "mix_class_group_margin_random": StrategySpec(
            "mix_class_group_margin_random",
            SchedulerConfig(
                mode="mix",
                mix={"class_group_balanced_entropy": 0.4, "margin": 0.3, "random": 0.3},
            ),
        ),
        "mix_interleaved_class_group_random": StrategySpec(
            "mix_interleaved_class_group_random",
            SchedulerConfig(mode="mix_interleaved", mix={"class_group_balanced_entropy": 0.7, "random": 0.3}),
        ),
        "mix_interleaved_class_group_margin_random": StrategySpec(
            "mix_interleaved_class_group_margin_random",
            SchedulerConfig(
                mode="mix_interleaved",
                mix={"class_group_balanced_entropy": 0.4, "margin": 0.3, "random": 0.3},
            ),
        ),
        "hybrid_weighted_entropy_coreset": StrategySpec(
            "hybrid_weighted_entropy_coreset",
            SchedulerConfig(
                mode="hybrid",
                hybrid={
                    "mode": "weighted",
                    "uncertainty": "entropy",
                    "diversity": "coreset_kcenter",
                    "uncertainty_weight": 0.5,
                    "diversity_weight": 0.5,
                },
            ),
        ),
        "hybrid_uncertainty_prefilter_coreset": StrategySpec(
            "hybrid_uncertainty_prefilter_coreset",
            SchedulerConfig(
                mode="hybrid",
                hybrid={
                    "mode": "uncertainty_prefilter_diversity",
                    "uncertainty": "entropy",
                    "diversity": "coreset_kcenter",
                    "prefilter_multiplier": 3.0,
                },
            ),
        ),
        "hybrid_diversity_prefilter_entropy": StrategySpec(
            "hybrid_diversity_prefilter_entropy",
            SchedulerConfig(
                mode="hybrid",
                hybrid={
                    "mode": "diversity_prefilter_uncertainty",
                    "uncertainty": "entropy",
                    "diversity": "coreset_kcenter",
                    "prefilter_multiplier": 3.0,
                },
            ),
        ),
        "hybrid_weighted_guarded": StrategySpec(
            "hybrid_weighted_guarded",
            SchedulerConfig(
                mode="hybrid",
                hybrid={
                    "mode": "weighted",
                    "uncertainty": "entropy",
                    "diversity": "coreset_kcenter",
                    "uncertainty_weight": 0.5,
                    "diversity_weight": 0.5,
                    "class_balance": True,
                    "group_balance": True,
                    "exploration_fraction": 0.2,
                },
            ),
        ),
    }


def choose_initial_seed(dataset: BenchmarkDataset, train_ids: Sequence[str], target_size: int, seed: int) -> list[str]:
    rng = random.Random(seed + 404)
    by_label: dict[str, list[str]] = defaultdict(list)
    sample_by_id = {sample.sample_id: sample for sample in dataset.samples}
    for sample_id in train_ids:
        by_label[sample_by_id[sample_id].label].append(sample_id)
    for ids in by_label.values():
        rng.shuffle(ids)

    label_order = list(dataset.labels)
    rng.shuffle(label_order)

    selected: list[str] = []
    for label in label_order:
        if by_label[label]:
            selected.append(by_label[label].pop())

    remaining = [sample_id for ids in by_label.values() for sample_id in ids]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, target_size - len(selected))])
    return selected[:target_size]


def compute_selection_diagnostics(
    selected_ids: Sequence[str],
    labeled_ids: Sequence[str],
    dataset: BenchmarkDataset,
    selected_embeddings: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    sample_by_id = {sample.sample_id: sample for sample in dataset.samples}
    selected_labels = Counter(sample_by_id[sample_id].label for sample_id in selected_ids)
    selected_groups = Counter(sample_by_id[sample_id].group_id for sample_id in selected_ids)
    cumulative_labels = Counter(sample_by_id[sample_id].label for sample_id in labeled_ids)
    duplicate_count = len(selected_ids) - len(set(selected_ids))
    group_total = sum(selected_groups.values())
    group_hhi = 0.0
    top_group_fraction = 0.0
    if group_total:
        group_hhi = sum((count / group_total) ** 2 for count in selected_groups.values())
        top_group_fraction = max(selected_groups.values()) / group_total
    redundancy = compute_embedding_redundancy_diagnostics(selected_embeddings)
    return {
        "selected_label_counts": dict(sorted(selected_labels.items())),
        "cumulative_label_counts": dict(sorted(cumulative_labels.items())),
        "selected_group_counts": dict(sorted(selected_groups.items())),
        "duplicate_selected_count": duplicate_count,
        "selected_duplicate_rate": duplicate_count / len(selected_ids) if selected_ids else 0.0,
        "selected_nn_distance_mean": redundancy["selected_nn_distance_mean"],
        "selected_nn_distance_min": redundancy["selected_nn_distance_min"],
        "top_group_fraction": top_group_fraction,
        "group_hhi": group_hhi,
    }


def compute_embedding_redundancy_diagnostics(
    embeddings: Sequence[Sequence[float]] | None,
) -> dict[str, float | None]:
    if embeddings is None or len(embeddings) < 2:
        return {"selected_nn_distance_mean": None, "selected_nn_distance_min": None}

    rows = _normalize_diagnostic_embeddings(embeddings)
    nearest_distances: list[float] = []
    for index, row in enumerate(rows):
        nearest_distances.append(
            min(
                _euclidean_distance(row, other)
                for other_index, other in enumerate(rows)
                if other_index != index
            )
        )

    return {
        "selected_nn_distance_mean": sum(nearest_distances) / len(nearest_distances),
        "selected_nn_distance_min": min(nearest_distances),
    }


def _normalize_diagnostic_embeddings(embeddings: Sequence[Sequence[float]]) -> list[list[float]]:
    rows: list[list[float]] = []
    expected_width: int | None = None
    for row_index, row in enumerate(embeddings):
        values = [float(value) for value in row]
        if not values:
            raise RuntimeError(f"Embedding diagnostics row {row_index} must not be empty.")
        if expected_width is None:
            expected_width = len(values)
        elif len(values) != expected_width:
            raise RuntimeError(
                f"Embedding diagnostics row {row_index} has width {len(values)}; expected {expected_width}."
            )
        if any(not math.isfinite(value) for value in values):
            raise RuntimeError(f"Embedding diagnostics row {row_index} contains a non-finite value.")
        rows.append(values)
    return rows


def _euclidean_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right)))


def compute_label_coverage_metrics(
    dataset: BenchmarkDataset,
    labeled_ids: Sequence[str],
    selected_ids: Sequence[str] = (),
) -> dict[str, float | int]:
    sample_by_id = {sample.sample_id: sample for sample in dataset.samples}
    covered_labels = {sample_by_id[sample_id].label for sample_id in labeled_ids}
    selected_labels = {sample_by_id[sample_id].label for sample_id in selected_ids}
    previously_covered = {sample_by_id[sample_id].label for sample_id in labeled_ids if sample_id not in set(selected_ids)}
    newly_selected_labels = selected_labels - previously_covered
    label_count = len(dataset.labels)

    test_support = Counter(sample.label for sample in dataset.samples if sample.split == "test")
    test_total = sum(test_support.values())
    missing_labels = set(dataset.labels) - covered_labels
    missing_test_support = sum(test_support[label] for label in missing_labels)

    return {
        "label_coverage_count": len(covered_labels),
        "label_coverage_fraction": len(covered_labels) / label_count if label_count else 0.0,
        "missing_label_count": len(missing_labels),
        "missing_label_fraction": len(missing_labels) / label_count if label_count else 0.0,
        "missing_test_support_weighted_fraction": missing_test_support / test_total if test_total else 0.0,
        "new_labels_selected_count": len(newly_selected_labels),
        "new_labels_selected_fraction": len(newly_selected_labels) / len(selected_labels) if selected_labels else 0.0,
    }


def compute_calibration_metrics(
    true_labels: Sequence[str],
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    ece_bins: int = 10,
) -> dict[str, float]:
    if ece_bins < 1:
        raise ValueError("ece_bins must be at least 1.")
    if len(true_labels) != len(probabilities):
        raise RuntimeError(
            f"Calibration input length mismatch: {len(true_labels)} labels and {len(probabilities)} probability rows."
        )
    if not labels:
        raise RuntimeError("Calibration metrics require at least one label.")

    label_index = {str(label): index for index, label in enumerate(labels)}
    if any(str(label) not in label_index for label in true_labels):
        unknown = sorted({str(label) for label in true_labels if str(label) not in label_index})
        raise RuntimeError(f"Calibration labels are not present in dataset.labels: {unknown}")

    clipped_true_probabilities: list[float] = []
    brier_values: list[float] = []
    bin_confidence_sums = [0.0 for _ in range(ece_bins)]
    bin_accuracy_sums = [0.0 for _ in range(ece_bins)]
    bin_counts = [0 for _ in range(ece_bins)]

    for row_index, (true_label, row) in enumerate(zip(true_labels, probabilities)):
        values = [float(value) for value in row]
        if len(values) != len(labels):
            raise RuntimeError(
                f"Calibration probability row {row_index} has width {len(values)}; expected {len(labels)}."
            )
        if any((not math.isfinite(value)) or value < 0.0 for value in values):
            raise RuntimeError(f"Calibration probability row {row_index} contains invalid probabilities.")

        row_sum = sum(values)
        if row_sum <= 0.0:
            raise RuntimeError(f"Calibration probability row {row_index} has non-positive mass.")
        if not math.isclose(row_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            values = [value / row_sum for value in values]

        true_index = label_index[str(true_label)]
        clipped_true_probabilities.append(max(values[true_index], 1e-15))
        brier_values.append(
            sum((probability - (1.0 if index == true_index else 0.0)) ** 2 for index, probability in enumerate(values))
        )

        predicted_index = max(range(len(values)), key=lambda index: values[index])
        confidence = values[predicted_index]
        bin_index = min(int(confidence * ece_bins), ece_bins - 1)
        bin_confidence_sums[bin_index] += confidence
        bin_accuracy_sums[bin_index] += 1.0 if predicted_index == true_index else 0.0
        bin_counts[bin_index] += 1

    sample_count = len(true_labels)
    if sample_count == 0:
        return {"multiclass_brier_score": 0.0, "nll": 0.0, "ece": 0.0}

    ece = 0.0
    for count, confidence_sum, accuracy_sum in zip(bin_counts, bin_confidence_sums, bin_accuracy_sums):
        if count == 0:
            continue
        ece += (count / sample_count) * abs((confidence_sum / count) - (accuracy_sum / count))

    return {
        "multiclass_brier_score": sum(brier_values) / sample_count,
        "nll": -sum(math.log(probability) for probability in clipped_true_probabilities) / sample_count,
        "ece": ece,
    }


def train_and_evaluate(
    model: SklearnTextBenchmarkAdapter,
    dataset: BenchmarkDataset,
    labeled_ids: Sequence[str],
    test_ids: Sequence[str],
) -> dict[str, float]:
    sample_by_id = {sample.sample_id: sample for sample in dataset.samples}
    model.fit(
        [sample_by_id[sample_id].text for sample_id in labeled_ids],
        [sample_by_id[sample_id].label for sample_id in labeled_ids],
    )
    test_texts = [sample_by_id[sample_id].text for sample_id in test_ids]
    test_labels = [sample_by_id[sample_id].label for sample_id in test_ids]
    metrics = model.evaluate(test_texts, test_labels)
    metrics.update(compute_calibration_metrics(test_labels, model.predict_proba(test_texts), dataset.labels))
    predictions = [str(value) for value in model.pipeline.predict(test_texts)] if model.pipeline is not None else []
    test_supported_labels = sorted(set(test_labels))
    per_label_recall = recall_score(
        test_labels,
        predictions,
        labels=test_supported_labels,
        average=None,
        zero_division=0,
    )
    zero_recall_count = sum(1 for value in per_label_recall if float(value) == 0.0)
    metrics["zero_recall_class_count"] = float(zero_recall_count)
    metrics["zero_recall_class_fraction"] = (
        float(zero_recall_count / len(test_supported_labels)) if test_supported_labels else 0.0
    )
    if dataset.rare_label is not None:
        metrics["rare_recall"] = float(
            recall_score(
                test_labels,
                predictions,
                labels=[dataset.rare_label],
                average="macro",
                zero_division=0,
            )
        )
    else:
        metrics["rare_recall"] = float("nan")
    return metrics


def run_one_curve(
    dataset: BenchmarkDataset,
    strategy: StrategySpec,
    budgets: Sequence[int],
    seed: int,
    initial_seed_size: int,
    budget_warnings: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_by_id = {sample.sample_id: sample for sample in dataset.samples}
    train_ids = sorted(sample.sample_id for sample in dataset.samples if sample.split == "train")
    test_ids = sorted(sample.sample_id for sample in dataset.samples if sample.split == "test")
    max_budget = min(max(budgets), len(train_ids))
    unique_budgets = sorted(set(budgets))
    usable_budgets = [budget for budget in unique_budgets if initial_seed_size <= budget <= max_budget]
    if budget_warnings is not None:
        for budget in unique_budgets:
            reason: str | None = None
            nearest_executable_budget: int | None = None
            if budget < initial_seed_size:
                reason = "below_initial_seed_size"
                nearest_executable_budget = initial_seed_size if initial_seed_size <= len(train_ids) else None
            elif budget > max_budget:
                reason = "above_train_pool_size"
                nearest_executable_budget = max_budget if max_budget >= initial_seed_size else None
            if reason is not None:
                budget_warnings.append(
                    {
                        "dataset": dataset.name,
                        "strategy": strategy.name,
                        "seed": seed,
                        "requested_budget": budget,
                        "reason": reason,
                        "initial_seed_size": initial_seed_size,
                        "train_pool_size": len(train_ids),
                        "nearest_executable_budget": nearest_executable_budget,
                    }
                )

    labeled_ids = choose_initial_seed(dataset, train_ids, initial_seed_size, seed)
    provider = InMemoryBenchmarkProvider([sample for sample in dataset.samples if sample.split == "train"])
    model = SklearnTextBenchmarkAdapter(dataset.labels, seed)
    scheduler = StrategyScheduler(strategy.scheduler_config)
    label_schema = LabelSchema(task="text_classification", labels=dataset.labels)

    metrics_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    current_metrics: dict[str, float] | None = None
    model_ready_for_current_labels = False

    for budget in usable_budgets:
        before_select = time.perf_counter()
        if not model_ready_for_current_labels:
            current_metrics = train_and_evaluate(model, dataset, labeled_ids, test_ids)
            model_ready_for_current_labels = True

        pool_ids = [sample_id for sample_id in train_ids if sample_id not in set(labeled_ids)]
        to_select = min(budget - len(labeled_ids), len(pool_ids))
        selected_ids: list[str] = []
        scheduler_snapshot: dict[str, Any] = {"mode": strategy.scheduler_config.mode, "no_acquisition_needed": True}
        if to_select > 0:
            context = SelectionContext(
                provider=provider,
                model=model,
                label_schema=label_schema,
                prediction_cache=None,
                embedding_cache=None,
                labeled_ids=labeled_ids,
                last_metrics={},
            )
            selected_ids, scheduler_snapshot = scheduler.select_batch(pool_ids, to_select, context, state={})
            selected_embeddings = (
                model.embed([sample_by_id[sample_id].text for sample_id in selected_ids]) if selected_ids else None
            )
            labeled_ids.extend(selected_ids)
            model_ready_for_current_labels = False
        else:
            selected_embeddings = None

        if not model_ready_for_current_labels:
            current_metrics = train_and_evaluate(model, dataset, labeled_ids, test_ids)
            model_ready_for_current_labels = True
        if current_metrics is None:
            raise RuntimeError("Benchmark curve did not produce evaluation metrics.")
        metrics = current_metrics
        elapsed_seconds = time.perf_counter() - before_select
        diagnostics = compute_selection_diagnostics(selected_ids, labeled_ids, dataset, selected_embeddings)
        coverage_metrics = compute_label_coverage_metrics(dataset, labeled_ids, selected_ids)
        rare_selected = (
            sum(1 for sample_id in selected_ids if sample_by_id[sample_id].label == dataset.rare_label)
            if dataset.rare_label is not None
            else 0
        )

        metrics_rows.append(
            {
                "dataset": dataset.name,
                "strategy": strategy.name,
                "seed": seed,
                "budget": len(labeled_ids),
                "requested_budget": budget,
                "initial_seed_size": initial_seed_size,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "macro_recall": metrics["macro_recall"],
                "rare_recall": metrics["rare_recall"],
                "zero_recall_class_count": metrics["zero_recall_class_count"],
                "zero_recall_class_fraction": metrics["zero_recall_class_fraction"],
                "multiclass_brier_score": metrics.get("multiclass_brier_score", float("nan")),
                "nll": metrics.get("nll", float("nan")),
                "ece": metrics.get("ece", float("nan")),
                "label_coverage_count": coverage_metrics["label_coverage_count"],
                "label_coverage_fraction": coverage_metrics["label_coverage_fraction"],
                "class_coverage_count": coverage_metrics["label_coverage_count"],
                "class_coverage_fraction": coverage_metrics["label_coverage_fraction"],
                "missing_label_count": coverage_metrics["missing_label_count"],
                "missing_label_fraction": coverage_metrics["missing_label_fraction"],
                "missing_class_count": coverage_metrics["missing_label_count"],
                "missing_class_fraction": coverage_metrics["missing_label_fraction"],
                "missing_test_support_weighted_fraction": coverage_metrics["missing_test_support_weighted_fraction"],
                "new_labels_selected_count": coverage_metrics["new_labels_selected_count"],
                "new_labels_selected_fraction": coverage_metrics["new_labels_selected_fraction"],
                "accuracy_per_label": metrics["accuracy"] / len(labeled_ids),
                "macro_f1_per_label": metrics["macro_f1"] / len(labeled_ids),
                "runtime_seconds": elapsed_seconds,
                "selected_count": len(selected_ids),
                "rare_selected_count": rare_selected,
            }
        )
        selection_rows.append(
            {
                "dataset": dataset.name,
                "strategy": strategy.name,
                "seed": seed,
                "budget": len(labeled_ids),
                "runtime_seconds": elapsed_seconds,
                "selected_ids": json.dumps(selected_ids, sort_keys=True),
                "scheduler_snapshot": json.dumps(scheduler_snapshot, sort_keys=True),
                "selected_label_counts": json.dumps(diagnostics["selected_label_counts"], sort_keys=True),
                "cumulative_label_counts": json.dumps(diagnostics["cumulative_label_counts"], sort_keys=True),
                "selected_group_counts": json.dumps(diagnostics["selected_group_counts"], sort_keys=True),
                "duplicate_selected_count": diagnostics["duplicate_selected_count"],
                "selected_duplicate_rate": diagnostics["selected_duplicate_rate"],
                "selected_nn_distance_mean": diagnostics["selected_nn_distance_mean"],
                "selected_nn_distance_min": diagnostics["selected_nn_distance_min"],
                "top_group_fraction": diagnostics["top_group_fraction"],
                "group_hhi": diagnostics["group_hhi"],
            }
        )

    return metrics_rows, selection_rows


def build_full_train_reference_row(dataset: BenchmarkDataset, seed: int) -> dict[str, Any]:
    train_ids = sorted(sample.sample_id for sample in dataset.samples if sample.split == "train")
    test_ids = sorted(sample.sample_id for sample in dataset.samples if sample.split == "test")
    model = SklearnTextBenchmarkAdapter(dataset.labels, seed)
    started = time.perf_counter()
    metrics = train_and_evaluate(model, dataset, train_ids, test_ids)
    elapsed_seconds = time.perf_counter() - started
    coverage_metrics = compute_label_coverage_metrics(dataset, train_ids)
    return {
        "dataset": dataset.name,
        "seed": seed,
        "train_size": len(train_ids),
        "test_size": len(test_ids),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "macro_recall": metrics["macro_recall"],
        "rare_recall": metrics["rare_recall"],
        "zero_recall_class_count": metrics["zero_recall_class_count"],
        "zero_recall_class_fraction": metrics["zero_recall_class_fraction"],
        "multiclass_brier_score": metrics.get("multiclass_brier_score", float("nan")),
        "nll": metrics.get("nll", float("nan")),
        "ece": metrics.get("ece", float("nan")),
        "label_coverage_count": coverage_metrics["label_coverage_count"],
        "label_coverage_fraction": coverage_metrics["label_coverage_fraction"],
        "class_coverage_count": coverage_metrics["label_coverage_count"],
        "class_coverage_fraction": coverage_metrics["label_coverage_fraction"],
        "missing_label_count": coverage_metrics["missing_label_count"],
        "missing_label_fraction": coverage_metrics["missing_label_fraction"],
        "missing_class_count": coverage_metrics["missing_label_count"],
        "missing_class_fraction": coverage_metrics["missing_label_fraction"],
        "missing_test_support_weighted_fraction": coverage_metrics["missing_test_support_weighted_fraction"],
        "runtime_seconds": elapsed_seconds,
    }


def add_curve_metrics(metrics_rows: list[dict[str, Any]]) -> None:
    by_curve: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics_rows:
        by_curve[(row["dataset"], row["strategy"], int(row["seed"]))].append(row)

    for rows in by_curve.values():
        rows.sort(key=lambda row: int(row["budget"]))
        for metric_name in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "macro_recall",
            "rare_recall",
            *CALIBRATION_COLUMNS,
        ]:
            if not all(metric_name in row for row in rows):
                continue
            values = [float(row[metric_name]) for row in rows]
            finite_pairs = [
                (float(row["budget"]), value)
                for row, value in zip(rows, values)
                if not math.isnan(value)
            ]
            aulc = normalized_auc(finite_pairs)
            for row in rows:
                row[f"aulc_{metric_name}"] = aulc

    random_index: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in metrics_rows:
        if row["strategy"] == "random":
            random_index[(row["dataset"], int(row["seed"]), int(row["budget"]))] = row

    for row in metrics_rows:
        baseline = random_index.get((row["dataset"], int(row["seed"]), int(row["budget"])))
        for metric_name in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "macro_recall",
            "rare_recall",
            *CALIBRATION_COLUMNS,
        ]:
            if metric_name not in row:
                continue
            current = float(row[metric_name])
            base = float(baseline[metric_name]) if baseline is not None and metric_name in baseline else float("nan")
            row[f"lift_{metric_name}_vs_random"] = current - base if not math.isnan(current) and not math.isnan(base) else float("nan")


def default_stop_policies() -> tuple[StopPolicySpec, ...]:
    return (
        StopPolicySpec(
            name="macro_f1_plateau_fast",
            policy_type="plateau",
            metric_name="macro_f1",
            min_budget=24,
            patience=1,
            min_delta=0.001,
        ),
        StopPolicySpec(
            name="macro_f1_plateau_conservative",
            policy_type="plateau",
            metric_name="macro_f1",
            min_budget=32,
            patience=2,
            min_delta=0.001,
        ),
        StopPolicySpec(
            name="accuracy_plateau_conservative",
            policy_type="plateau",
            metric_name="accuracy",
            min_budget=32,
            patience=2,
            min_delta=0.001,
        ),
    )


def stop_policy_spec_to_dict(policy: StopPolicySpec) -> dict[str, Any]:
    return {
        "name": policy.name,
        "policy_type": policy.policy_type,
        "metric_name": policy.metric_name,
        "min_budget": policy.min_budget,
        "patience": policy.patience,
        "min_delta": policy.min_delta,
    }


def simulate_stop_policies(
    metrics_rows: Sequence[dict[str, Any]],
    policies: Sequence[StopPolicySpec] | None = None,
) -> list[dict[str, Any]]:
    policy_specs = tuple(policies or default_stop_policies())
    by_curve: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics_rows:
        by_curve[(str(row["dataset"]), str(row["strategy"]), int(row["seed"]))].append(row)

    stop_rows: list[dict[str, Any]] = []
    for curve_key, curve_rows in sorted(by_curve.items()):
        ordered_rows = sorted(curve_rows, key=lambda row: int(row["budget"]))
        for policy in policy_specs:
            stop_rows.append(_simulate_stop_policy_for_curve(curve_key, ordered_rows, policy))
    return stop_rows


def _simulate_stop_policy_for_curve(
    curve_key: tuple[str, str, int],
    curve_rows: Sequence[dict[str, Any]],
    policy: StopPolicySpec,
) -> dict[str, Any]:
    if not curve_rows:
        raise RuntimeError("Cannot simulate a stop policy for an empty benchmark curve.")
    if policy.policy_type != "plateau":
        raise ValueError(f"Unsupported benchmark stop policy type: {policy.policy_type}")
    if policy.metric_name not in {"macro_f1", "accuracy", "weighted_f1", "rare_recall"}:
        raise ValueError(f"Unsupported benchmark stop metric: {policy.metric_name}")
    if policy.patience < 1:
        raise ValueError("Stop policy patience must be at least 1.")

    final_row = curve_rows[-1]
    stop_row, stop_reason = _find_plateau_stop_row(curve_rows, policy)
    if stop_row is None:
        stop_row = final_row
        stop_reason = "final_budget_no_stop"

    dataset, strategy, seed = curve_key
    stopped_budget = int(stop_row["budget"])
    full_budget = int(final_row["budget"])
    stop_metric = _finite_float_or_none(stop_row.get(policy.metric_name))
    full_metric = _finite_float_or_none(final_row.get(policy.metric_name))
    runtime_at_stop = _cumulative_runtime_at_budget(curve_rows, stopped_budget)
    runtime_full = _cumulative_runtime_at_budget(curve_rows, full_budget)

    return {
        "dataset": dataset,
        "strategy": strategy,
        "seed": seed,
        "policy_name": policy.name,
        "policy_type": policy.policy_type,
        "metric_name": policy.metric_name,
        "policy_min_budget": policy.min_budget,
        "policy_patience": policy.patience,
        "policy_min_delta": policy.min_delta,
        "stop_reason": stop_reason,
        "stopped_budget": stopped_budget,
        "full_budget": full_budget,
        "final_budget": full_budget,
        "labels_saved": max(0, full_budget - stopped_budget),
        "relative_savings": ((full_budget - stopped_budget) / full_budget) if full_budget > 0 else None,
        "stop_metric": stop_metric,
        "full_metric": full_metric,
        "quality_delta_vs_full": (
            stop_metric - full_metric if stop_metric is not None and full_metric is not None else None
        ),
        "runtime_at_stop": runtime_at_stop,
        "runtime_full": runtime_full,
        "runtime_saved": (
            runtime_full - runtime_at_stop if runtime_at_stop is not None and runtime_full is not None else None
        ),
    }


def _find_plateau_stop_row(
    curve_rows: Sequence[dict[str, Any]],
    policy: StopPolicySpec,
) -> tuple[dict[str, Any] | None, str | None]:
    best_value: float | None = None
    stale_count = 0
    for row in curve_rows:
        value = _finite_float_or_none(row.get(policy.metric_name))
        if value is None:
            continue

        if best_value is None or value > best_value + policy.min_delta:
            best_value = value
            stale_count = 0
        else:
            stale_count += 1

        if int(row["budget"]) >= policy.min_budget and stale_count >= policy.patience:
            return row, f"{policy.metric_name}_plateau_patience_{policy.patience}"
    return None, None


def _finite_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _cumulative_runtime_at_budget(curve_rows: Sequence[dict[str, Any]], budget: int) -> float | None:
    total = 0.0
    saw_runtime = False
    for row in curve_rows:
        if int(row["budget"]) > budget:
            break
        runtime = _finite_float_or_none(row.get("runtime_seconds"))
        if runtime is None:
            continue
        total += runtime
        saw_runtime = True
    return total if saw_runtime else None


def summarize_stop_policy_rows(stop_policy_rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stop_policy_rows:
        by_policy[str(row["policy_name"])].append(row)

    summary: dict[str, dict[str, Any]] = {}
    for policy_name, rows in sorted(by_policy.items()):
        savings = [
            value
            for value in (_finite_float_or_none(row.get("relative_savings")) for row in rows)
            if value is not None
        ]
        labels_saved = [
            value
            for value in (_finite_float_or_none(row.get("labels_saved")) for row in rows)
            if value is not None
        ]
        deltas = [
            value
            for value in (_finite_float_or_none(row.get("quality_delta_vs_full")) for row in rows)
            if value is not None
        ]
        summary[policy_name] = {
            "policy_type": rows[0]["policy_type"],
            "metric_name": rows[0]["metric_name"],
            "row_count": len(rows),
            "stopped_count": sum(1 for row in rows if row["stop_reason"] != "final_budget_no_stop"),
            "mean_labels_saved": sum(labels_saved) / len(labels_saved) if labels_saved else None,
            "mean_relative_savings": sum(savings) / len(savings) if savings else None,
            "mean_quality_delta_vs_full": sum(deltas) / len(deltas) if deltas else None,
        }
    return summary


def normalized_auc(points: Sequence[tuple[float, float]]) -> float:
    if not points:
        return float("nan")
    if len(points) == 1:
        return points[0][1]
    ordered = sorted(points)
    min_x = ordered[0][0]
    max_x = ordered[-1][0]
    if math.isclose(min_x, max_x):
        return ordered[-1][1]
    area = 0.0
    for (x0, y0), (x1, y1) in zip(ordered, ordered[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area / (max_x - min_x)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                key: ("" if isinstance(value, float) and not math.isfinite(value) else value)
                for key, value in row.items()
            }
            for row in rows
        )


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(item) for item in value]
    return value


def write_strict_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_json_value(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _run_git_command(args: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def collect_reproducibility_metadata(argv: Sequence[str]) -> dict[str, Any]:
    status_output = _run_git_command(["status", "--porcelain"])
    status_count = None if status_output is None else len(status_output.splitlines())
    return {
        "artifact_schema_version": BENCHMARK_ARTIFACT_SCHEMA_VERSION,
        "argv": list(argv),
        "git": {
            "sha": _run_git_command(["rev-parse", "HEAD"]),
            "dirty": None if status_output is None else bool(status_output),
            "status_entry_count": status_count,
            "status_count": status_count,
        },
        "runtime": {
            "python_version": sys.version,
            "python_version_info": {
                "major": sys.version_info.major,
                "minor": sys.version_info.minor,
                "micro": sys.version_info.micro,
            },
            "platform": platform.platform(),
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
        },
    }


def default_output_dir_for_preset(preset: str, run_id: str) -> Path:
    return Path("benchmarks") / "results" / preset / run_id


def make_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Benchmark output path exists and is not a directory: {output_dir}")
    existing_entries = sorted(output_dir.iterdir(), key=lambda path: path.name) if output_dir.exists() else []
    if existing_entries and not overwrite:
        preview = ", ".join(path.name for path in existing_entries[:5])
        if len(existing_entries) > 5:
            preview = f"{preview}, ..."
        raise FileExistsError(
            f"Refusing to write benchmark artifacts into non-empty output directory {output_dir}: {preview}. "
            "Use --overwrite for intentional replacement or choose a fresh output directory."
        )
    if existing_entries and overwrite:
        for entry in existing_entries:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _format_optional_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def write_summary(
    output_dir: Path,
    metrics_rows: Sequence[dict[str, Any]],
    selection_rows: Sequence[dict[str, Any]],
    stop_policy_rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    full_reference_rows: Sequence[dict[str, Any]] = (),
) -> None:
    best_by_dataset: dict[str, dict[str, Any]] = {}
    for row in metrics_rows:
        dataset = str(row["dataset"])
        current = best_by_dataset.get(dataset)
        if current is None or float(row["macro_f1"]) > float(current["macro_f1"]):
            best_by_dataset[dataset] = row

    summary = {
        "manifest": manifest,
        "best_macro_f1_by_dataset": best_by_dataset,
        "stop_policy_rows": stop_policy_rows,
        "stop_policy_summary": summarize_stop_policy_rows(stop_policy_rows),
        "full_train_reference_rows": full_reference_rows,
        "row_counts": {
            "metrics": len(metrics_rows),
            "selections": len(selection_rows),
            "stop_policies": len(stop_policy_rows),
            "full_train_reference": len(full_reference_rows),
        },
    }
    write_strict_json(output_dir / "summary.json", summary)
    stop_policy_summary = summary["stop_policy_summary"]

    lines = [
        "# SDK-First Benchmark Summary",
        "",
        f"- Run id: `{manifest['run_id']}`",
        f"- Datasets: {', '.join(manifest['datasets'])}",
        f"- Strategies: {', '.join(manifest['strategies'])}",
        f"- Budgets: {', '.join(str(value) for value in manifest['budgets'])}",
        f"- Seeds: {', '.join(str(value) for value in manifest['seeds'])}",
        "",
        "## Best Macro-F1 By Dataset",
        "",
        "| Dataset | Strategy | Seed | Budget | Macro-F1 | Accuracy |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset, row in sorted(best_by_dataset.items()):
        lines.append(
            f"| {dataset} | {row['strategy']} | {row['seed']} | {row['budget']} | "
            f"{float(row['macro_f1']):.4f} | {float(row['accuracy']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Stop Policy Diagnostics",
            "",
            "| Policy | Metric | Curves | Stops | Mean Label Savings | Mean Quality Delta |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for policy_name, policy_summary in sorted(stop_policy_summary.items()):
        mean_savings = _finite_float_or_none(policy_summary.get("mean_relative_savings"))
        mean_delta = _finite_float_or_none(policy_summary.get("mean_quality_delta_vs_full"))
        lines.append(
            f"| {policy_name} | {policy_summary['metric_name']} | {policy_summary['row_count']} | "
            f"{policy_summary['stopped_count']} | "
            f"{_format_optional_float(mean_savings)} | {_format_optional_float(mean_delta)} |"
        )
    lines.extend(
        [
            "",
            "Artifacts in this directory:",
            "",
            "- `metrics.csv`: budgeted quality metrics, AULC, lift versus random, runtime, and budget efficiency columns.",
            "- `selections.csv`: selected ids, scheduler snapshots, label mix, duplicate counts, and group concentration diagnostics.",
            "- `stop_policies.csv`: post-hoc stop policy decisions with label savings, quality deltas, and runtime savings.",
            "- `full_train_reference.csv`: no-budget reference metrics from fitting on the full train split.",
            "- `budget_warnings.csv`: requested budgets that were not executable, with explicit skip reasons.",
            "- `manifest.json`: run configuration and SDK gap notes.",
            "- `summary.json`: machine-readable rollup.",
            "- `validation.json`: acquisition-surface checks for opaque ids, groups, schema, and metadata.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _labels_for_ids(dataset: BenchmarkDataset, sample_ids: Sequence[str]) -> dict[str, str]:
    sample_by_id = {sample.sample_id: sample for sample in dataset.samples}
    return {sample_id: sample_by_id[sample_id].label for sample_id in sample_ids}


def _sample_ids_by_split(dataset: BenchmarkDataset, split: str) -> list[str]:
    return sorted(sample.sample_id for sample in dataset.samples if sample.split == split)


def _run_public_project_round(project: ActiveLearningProject, batch_size: int) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for _ in range(12):
        result = project.run_step(batch_size=batch_size, poll_interval_seconds=0)
        steps.append(
            {
                "step": result.step.value if hasattr(result.step, "value") else str(result.step),
                "round_id": result.round_id,
                "message": result.message,
                "details": result.details,
            }
        )
        rounds = [project.get_round(item["round_id"]) for item in project.list_rounds()]
        if rounds and rounds[-1].get("status") == "done" and rounds[-1].get("selected_sample_ids"):
            return steps
    raise RuntimeError("Project smoke did not complete an active round within 12 run_step calls.")


def _validate_project_smoke_steps(steps: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not steps:
        raise RuntimeError("Project smoke did not record any public run_step calls.")

    seed_step = steps[0]
    if seed_step.get("step") != "train_eval":
        raise RuntimeError(f"Expected first project smoke step to be seed train_eval, got: {seed_step}")
    if seed_step.get("round_id") is not None:
        raise RuntimeError(f"Expected seed train_eval round_id to be None, got: {seed_step.get('round_id')!r}")
    if seed_step.get("details", {}).get("seed") is not True:
        raise RuntimeError(f"Expected first train_eval to be marked as seed step, got: {seed_step}")

    active_steps = [step for step in steps if step.get("round_id") is not None]
    active_sequence = [str(step.get("step")) for step in active_steps]
    expected_active_sequence = ["select", "push", "wait", "pull", "train_eval", "update"]
    if active_sequence != expected_active_sequence:
        raise RuntimeError(
            "Project smoke active round sequence mismatch: "
            f"expected {expected_active_sequence}, got {active_sequence}"
        )

    active_round_ids = sorted({str(step["round_id"]) for step in active_steps})
    if len(active_round_ids) != 1:
        raise RuntimeError(f"Expected exactly one active round id, got: {active_round_ids}")

    return {
        "seed_train_step_observed": True,
        "seed_train_round_id_is_none": True,
        "active_round_step_sequence": active_sequence,
        "active_round_id": active_round_ids[0],
    }


def run_project_smoke(
    *,
    output_dir: Path,
    seed: int,
    initial_seed_size: int,
    batch_size: int,
    dataset_name: str = "separable_topics",
    strategy_name: str = "entropy",
    overwrite: bool = False,
    argv: Sequence[str] = (),
    run_id: str | None = None,
) -> None:
    run_id = run_id or make_run_id()
    started = time.perf_counter()
    prepare_output_dir(output_dir, overwrite=overwrite)

    dataset = DATASET_BUILDERS[dataset_name](seed)
    validation_report = validate_acquisition_surface(dataset)
    train_ids = _sample_ids_by_split(dataset, "train")
    test_ids = _sample_ids_by_split(dataset, "test")
    initial_seed_ids = choose_initial_seed(dataset, train_ids, initial_seed_size, seed)
    validation_ids = choose_initial_seed(dataset, test_ids, len(dataset.labels) * 3, seed + 707)
    project_samples = [
        sample
        for sample in dataset.samples
        if sample.sample_id in set(train_ids).union(validation_ids)
    ]
    provider = InMemoryBenchmarkProvider(project_samples)
    label_schema = LabelSchema(task="text_classification", labels=dataset.labels)
    label_by_id = _labels_for_ids(dataset, [sample.sample_id for sample in project_samples])
    backend = OracleProjectSmokeBackend(label_by_id)
    model = SklearnTextClassifierAdapter(
        estimator=Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                (
                    "classifier",
                    LogisticRegression(
                        C=2.0,
                        max_iter=500,
                        random_state=seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    )

    workdir = output_dir / "workdir"
    if workdir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Refusing to replace existing project-smoke workdir in {output_dir}. "
                "Use --overwrite for intentional replacement or choose a fresh output directory."
            )
        shutil.rmtree(workdir)

    project = ActiveLearningProject("benchmark-project-smoke", workdir=workdir)
    try:
        project.configure(
            dataset=provider,
            model=model,
            label_schema=label_schema,
            label_backend_config=LabelBackendConfig(backend="custom"),
            label_backend=backend,
            scheduler_config=strategy_specs()[strategy_name].scheduler_config,
            annotation_policy=AnnotationPolicy(mode="latest", min_votes=1),
            cache_config=CacheConfig(enable=False, persist=False),
            fingerprint_config=FingerprintConfig(mode="strict"),
            split_config=SplitConfig(
                mode="explicit",
                explicit_splits={"train": train_ids, "val": validation_ids, "test": []},
            ),
        )

        train_import = project.import_labels(
            _labels_for_ids(dataset, initial_seed_ids),
            source="project-smoke-initial-train-seed",
        )
        validation_import = project.import_labels(
            _labels_for_ids(dataset, validation_ids),
            source="project-smoke-validation-labels",
        )
        status_after_import = project.status()
        model_id_before_public_steps = model.get_model_id()

        steps = _run_public_project_round(project, batch_size)
        step_validation = _validate_project_smoke_steps(steps)
        final_status = project.status()
        rounds = [project.get_round(item["round_id"]) for item in project.list_rounds()]
        project_validation = project.validate()
        metrics_history = [
            {"step": record.step, "metrics": dict(record.metrics)}
            for record in project.get_state().metrics_history
        ]
        model_id_after_public_steps = model.get_model_id()
    finally:
        project.close()

    completed_rounds = [round_info for round_info in rounds if round_info.get("status") == "done"]
    if not completed_rounds:
        raise RuntimeError("Project smoke completed without a DONE round.")
    active_round = completed_rounds[-1]
    selected_ids = list(active_round.get("selected_sample_ids", []))
    if not selected_ids:
        raise RuntimeError("Project smoke completed a round without selected samples.")
    forbidden_selection = set(initial_seed_ids).union(validation_ids)
    selected_forbidden = sorted(set(selected_ids).intersection(forbidden_selection))
    if selected_forbidden:
        raise RuntimeError(f"Project smoke selected already imported labels: {selected_forbidden}")
    if not project_validation.get("ok"):
        raise RuntimeError(f"Project validation failed: {project_validation}")
    if not active_round.get("metrics_after"):
        raise RuntimeError("Project smoke completed an active round without metrics after training.")
    if model_id_before_public_steps == model_id_after_public_steps:
        raise RuntimeError("Project smoke model id did not change after public run_step training.")
    metric_steps = [record["step"] for record in metrics_history]
    if metric_steps[:2] != ["seed_eval", "eval"]:
        raise RuntimeError(f"Expected seed and active evaluation metric records, got: {metric_steps}")
    if any(not record["metrics"] for record in metrics_history[:2]):
        raise RuntimeError(f"Expected non-empty seed and active evaluation metrics, got: {metrics_history[:2]}")

    selected_labels = Counter(label_by_id[sample_id] for sample_id in selected_ids)
    final_train_labeled_ids = [
        sample_id
        for sample_id in train_ids
        if sample_id in set(initial_seed_ids).union(selected_ids)
    ]
    artifact_names = {
        "summary_json": "summary.json",
        "summary_md": "summary.md",
        "manifest_json": "manifest.json",
        "workdir_state": "workdir/state.json",
    }
    manifest = {
        **collect_reproducibility_metadata(argv),
        "run_id": run_id,
        "preset": "project_smoke",
        "benchmark_claim_category": CLAIM_CATEGORY_END_TO_END_PUBLIC_PROJECT_WORKFLOW,
        "benchmark_contract": (
            "Exercises the public ActiveLearningProject facade for seed label import and one active round through "
            "select, backend push, wait, pull, train/eval, and state update using a benchmark-only oracle backend. "
            "This proves public workflow wiring and reproducibility metadata for the smoke fixture; it does not "
            "measure external human-labeling services or establish active-learning quality superiority."
        ),
        "dataset": dataset.name,
        "strategy": strategy_name,
        "seed": seed,
        "initial_seed_size": len(initial_seed_ids),
        "batch_size": batch_size,
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": artifact_names,
        "artifact_names": sorted(set(artifact_names.values())),
    }

    summary = {
        "run_id": run_id,
        "dataset": dataset.name,
        "strategy": strategy_name,
        "seed": seed,
        "initial_seed_size": len(initial_seed_ids),
        "validation_label_count": len(validation_ids),
        "batch_size": batch_size,
        "public_facade_calls": [
            "ActiveLearningProject.configure",
            "ActiveLearningProject.import_labels",
            "ActiveLearningProject.run_step",
            "ActiveLearningProject.status",
            "ActiveLearningProject.validate",
        ],
        "model_adapter": {
            "source": "active_learning_sdk.adapters",
            "class": "SklearnTextClassifierAdapter",
            "qualified_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
            "estimator_class": f"{model.estimator.__class__.__module__}.{model.estimator.__class__.__qualname__}",
        },
        "private_state_mutation_used": False,
        "model_warm_started_from_imported_seed": False,
        "model_id_before_public_steps": model_id_before_public_steps,
        "model_id_after_public_steps": model_id_after_public_steps,
        "import_summaries": {
            "train_seed": train_import,
            "validation": validation_import,
        },
        "status_after_import": status_after_import,
        "final_status": final_status,
        "rounds": rounds,
        "steps": steps,
        "selected_ids": selected_ids,
        "selected_label_counts": dict(sorted(selected_labels.items())),
        "final_train_labeled_count": len(final_train_labeled_ids),
        "validation": {
            "project_validate": project_validation,
            "acquisition_surface": validation_report,
            "strict_json": True,
            "public_step_sequence": step_validation,
            "metrics_history": metrics_history,
            "model_id_changed_after_public_steps": model_id_before_public_steps != model_id_after_public_steps,
            "active_round_has_metrics_after_training": bool(active_round.get("metrics_after")),
            "selected_previously_imported_count": 0,
            "backend_pushed_sample_ids": backend.pushed_sample_ids,
            "backend_pulled_sample_ids": backend.pulled_sample_ids,
            "backend_labels_only_after_push": sorted(backend.pushed_sample_ids) == sorted(selected_ids),
            "completed_active_rounds": len(completed_rounds),
        },
        "artifact_paths": {
            "summary_json": str((output_dir / "summary.json").as_posix()),
            "summary_md": str((output_dir / "summary.md").as_posix()),
            "manifest_json": str((output_dir / "manifest.json").as_posix()),
            "workdir_state": str((workdir / "state.json").as_posix()),
        },
        "manifest": manifest,
    }
    if not summary["validation"]["backend_labels_only_after_push"]:
        raise RuntimeError("Oracle backend did not restrict labels to pushed samples.")

    write_strict_json(output_dir / "summary.json", summary)
    write_strict_json(output_dir / "manifest.json", manifest)
    lines = [
        "# Full Project Smoke Summary",
        "",
        f"- Dataset: `{dataset.name}`",
        f"- Strategy: `{strategy_name}`",
        f"- Initial train seed labels imported: {len(initial_seed_ids)}",
        f"- Validation labels imported: {len(validation_ids)}",
        f"- Selected in active round: {len(selected_ids)}",
        f"- Final train labeled count: {len(final_train_labeled_ids)}",
        f"- Project validation ok: {project_validation.get('ok')}",
        f"- Private state mutation used: {summary['private_state_mutation_used']}",
        f"- External model warm-start used: {summary['model_warm_started_from_imported_seed']}",
        f"- Model adapter: `{summary['model_adapter']['source']}.{summary['model_adapter']['class']}`",
        f"- Model id changed after public steps: {summary['validation']['model_id_changed_after_public_steps']}",
        f"- Seed train step observed: {step_validation['seed_train_step_observed']}",
        "",
        "## Selected Label Counts",
        "",
        *[f"- `{label}`: {count}" for label, count in sorted(selected_labels.items())],
        "",
        "## Public Loop Steps",
        "",
        *[f"- `{step['step']}`: {step['message']}" for step in steps],
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_csv_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def default_datasets_for_preset(preset: str) -> list[str]:
    if preset == "smoke":
        return ["separable_topics", "rare_class_trap"]
    if preset == "full":
        return list(SYNTHETIC_DATASET_NAMES)
    if preset == "real_smoke":
        return ["banking77"]
    if preset == "real_medium":
        return ["banking77", "clinc_oos_imbalanced"]
    if preset == "real_full":
        return list(REAL_DATASET_NAMES)
    if preset == "project_smoke":
        return ["separable_topics"]
    raise ValueError(f"Unsupported preset: {preset}")


def default_strategies_for_preset(preset: str, available_strategies: dict[str, StrategySpec]) -> list[str]:
    if preset in {"smoke", "real_smoke"}:
        return ["random", "entropy", "margin", "least_confidence", "mix_entropy_random"]
    if preset == "real_medium":
        return [
            "random",
            "entropy",
            "margin",
            "least_confidence",
            "embedding_kmeans_pp",
            "class_group_balanced_entropy",
            "coreset_kcenter",
            "badge",
            "mix_class_group_random",
        ]
    if preset in {"full", "real_full"}:
        return list(available_strategies.keys())
    if preset == "project_smoke":
        return ["entropy"]
    raise ValueError(f"Unsupported preset: {preset}")


def default_budgets_for_preset(preset: str) -> list[int]:
    if preset == "smoke":
        return list(SMOKE_BUDGETS)
    if preset in {"real_smoke", "real_medium", "real_full"}:
        return list(REAL_DATASET_BUDGETS)
    return list(DEFAULT_BUDGETS)


def initial_seed_class_count_is_valid(dataset: BenchmarkDataset, initial_seed_ids: Sequence[str]) -> bool:
    sample_by_id = {sample.sample_id: sample for sample in dataset.samples}
    return len({sample_by_id[sample_id].label for sample_id in initial_seed_ids}) >= 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic SDK-first active-learning benchmarks on synthetic or opt-in real text datasets."
    )
    parser.add_argument(
        "--preset",
        choices=["smoke", "full", "project_smoke", "real_smoke", "real_medium", "real_full"],
        default="smoke",
    )
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names. Defaults depend on preset.")
    parser.add_argument("--strategies", default=None, help="Comma-separated strategy names. Defaults depend on preset.")
    parser.add_argument("--budgets", default=None, help="Comma-separated cumulative label budgets.")
    parser.add_argument("--seeds", default="13", help="Comma-separated integer seeds.")
    parser.add_argument("--initial-seed-size", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=6, help="Project-smoke active round batch size.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional deterministic cap for real train pools.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional deterministic cap for real test splits.")
    parser.add_argument(
        "--allow-uncapped-real-standard",
        action="store_true",
        help=(
            "Allow real_medium/real_full without train/test caps for local exploratory runs. "
            "Such output is marked local_uncapped_override and is not Stage 11 standard evidence."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty benchmark output directory.")
    return parser


def real_evidence_level(
    preset: str,
    seed_count: int,
    *,
    max_train_samples: int | None,
    max_test_samples: int | None,
    allow_uncapped_real_standard: bool,
) -> str:
    if preset == "real_smoke":
        return "smoke_only"
    if preset not in REAL_STANDARD_PRESETS:
        return "synthetic_or_project"
    if allow_uncapped_real_standard:
        return "local_uncapped_override"
    if (
        max_train_samples is not None
        and max_train_samples > 0
        and max_test_samples is not None
        and max_test_samples > 0
        and seed_count >= REAL_STANDARD_MIN_SEEDS
    ):
        return "standard"
    return "invalid_standard_request"


def validate_real_standard_request(args: argparse.Namespace, seeds: Sequence[int]) -> None:
    if args.preset not in REAL_STANDARD_PRESETS:
        return
    if len(set(seeds)) < REAL_STANDARD_MIN_SEEDS:
        raise SystemExit(
            f"--preset {args.preset} is Stage 11 standard real evidence and requires at least "
            f"{REAL_STANDARD_MIN_SEEDS} distinct seeds, for example --seeds 13,21,34."
        )
    if args.allow_uncapped_real_standard:
        return
    if args.max_train_samples is None or args.max_train_samples <= 0:
        raise SystemExit(
            f"--preset {args.preset} requires an explicit positive --max-train-samples cap "
            "unless --allow-uncapped-real-standard is supplied for local exploratory runs."
        )
    if args.max_test_samples is None or args.max_test_samples <= 0:
        raise SystemExit(
            f"--preset {args.preset} requires an explicit positive --max-test-samples cap "
            "unless --allow-uncapped-real-standard is supplied for local exploratory runs."
        )


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    ensure_benchmark_dependencies()
    available_strategies = strategy_specs()
    run_id = make_run_id()

    if args.preset == "project_smoke":
        seeds = parse_int_list(args.seeds)
        if len(seeds) != 1:
            raise SystemExit("--preset project_smoke expects exactly one seed.")
        datasets = parse_csv_list(args.datasets) if args.datasets else default_datasets_for_preset(args.preset)
        strategies = parse_csv_list(args.strategies) if args.strategies else default_strategies_for_preset(args.preset, available_strategies)
        if len(datasets) != 1:
            raise SystemExit("--preset project_smoke expects exactly one dataset.")
        if len(strategies) != 1:
            raise SystemExit("--preset project_smoke expects exactly one strategy.")
        if datasets[0] not in DATASET_BUILDERS:
            raise SystemExit(f"Unknown dataset: {datasets[0]}")
        if strategies[0] not in available_strategies:
            raise SystemExit(f"Unknown strategy: {strategies[0]}")
        initial_dataset = build_benchmark_dataset(
            datasets[0],
            seeds[0],
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
        )
        initial_train_ids = _sample_ids_by_split(initial_dataset, "train")
        if not initial_seed_class_count_is_valid(
            initial_dataset,
            choose_initial_seed(initial_dataset, initial_train_ids, args.initial_seed_size, seeds[0]),
        ):
            raise SystemExit("--initial-seed-size must include at least two classes for sklearn training.")
        output_dir = Path(args.output_dir) if args.output_dir else default_output_dir_for_preset(args.preset, run_id)
        run_project_smoke(
            output_dir=output_dir,
            seed=seeds[0],
            initial_seed_size=args.initial_seed_size,
            batch_size=args.batch_size,
            dataset_name=datasets[0],
            strategy_name=strategies[0],
            overwrite=args.overwrite,
            argv=raw_argv,
            run_id=run_id,
        )
        print(f"Wrote project smoke artifacts to {output_dir.resolve()}")
        return 0

    datasets = (
        parse_csv_list(args.datasets) if args.datasets else default_datasets_for_preset(args.preset)
    )
    strategies = (
        parse_csv_list(args.strategies)
        if args.strategies
        else default_strategies_for_preset(args.preset, available_strategies)
    )
    budgets = parse_int_list(args.budgets) if args.budgets else default_budgets_for_preset(args.preset)
    seeds = parse_int_list(args.seeds)

    unknown_datasets = sorted(set(datasets) - set(DATASET_BUILDERS))
    unknown_strategies = sorted(set(strategies) - set(available_strategies))
    if unknown_datasets:
        raise SystemExit(f"Unknown datasets: {', '.join(unknown_datasets)}")
    if unknown_strategies:
        raise SystemExit(f"Unknown strategies: {', '.join(unknown_strategies)}")
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer seed.")
    validate_real_standard_request(args, seeds)
    initial_seed_datasets = [
        build_benchmark_dataset(
            name,
            seeds[0],
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
        )
        for name in datasets
    ]
    for dataset in initial_seed_datasets:
        train_ids = _sample_ids_by_split(dataset, "train")
        initial_seed_ids = choose_initial_seed(dataset, train_ids, args.initial_seed_size, seeds[0])
        if dataset.name in REAL_DATASET_REGISTRY:
            if not initial_seed_class_count_is_valid(dataset, initial_seed_ids):
                raise SystemExit("--initial-seed-size must include at least two classes for sklearn training.")
        elif args.initial_seed_size < len(dataset.labels):
            raise SystemExit("--initial-seed-size must be large enough to include at least one sample per class.")

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir_for_preset(args.preset, run_id)
    prepare_output_dir(output_dir, overwrite=args.overwrite)
    started = time.perf_counter()
    metrics_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    budget_warning_rows: list[dict[str, Any]] = []
    full_reference_rows: list[dict[str, Any]] = []

    for seed in seeds:
        for dataset_name in datasets:
            dataset = build_benchmark_dataset(
                dataset_name,
                seed,
                max_train_samples=args.max_train_samples,
                max_test_samples=args.max_test_samples,
            )
            validation_rows.append(validate_acquisition_surface(dataset))
            full_reference_rows.append(build_full_train_reference_row(dataset, seed))
            for strategy_name in strategies:
                curve_metrics, curve_selections = run_one_curve(
                    dataset,
                    available_strategies[strategy_name],
                    budgets,
                    seed,
                    args.initial_seed_size,
                    budget_warning_rows,
                )
                metrics_rows.extend(curve_metrics)
                selection_rows.extend(curve_selections)

    add_curve_metrics(metrics_rows)
    metrics_rows.sort(key=lambda row: (row["dataset"], row["seed"], row["strategy"], int(row["budget"])))
    selection_rows.sort(key=lambda row: (row["dataset"], row["seed"], row["strategy"], int(row["budget"])))
    stop_policy_specs = default_stop_policies()
    stop_policy_rows = simulate_stop_policies(metrics_rows, stop_policy_specs)
    stop_policy_rows.sort(
        key=lambda row: (
            row["dataset"],
            row["seed"],
            row["strategy"],
            row["policy_name"],
        )
    )

    artifact_names = {
        "metrics_csv": "metrics.csv",
        "selections_csv": "selections.csv",
        "stop_policies_csv": "stop_policies.csv",
        "full_train_reference_csv": "full_train_reference.csv",
        "budget_warnings_csv": "budget_warnings.csv",
        "manifest_json": "manifest.json",
        "summary_json": "summary.json",
        "summary_md": "summary.md",
        "validation_json": "validation.json",
    }
    manifest = {
        **collect_reproducibility_metadata(raw_argv),
        "run_id": run_id,
        "preset": args.preset,
        "benchmark_claim_category": CLAIM_CATEGORY_ACTIVE_LEARNING_QUALITY,
        "benchmark_contract": (
            "Compares SDK acquisition strategies under the same benchmark-owned datasets, sklearn text adapter, "
            "pool order, seeds, and cumulative label budgets. Acquisition calls use StrategyScheduler and "
            "SelectionContext, while true labels remain hidden from the acquisition-visible provider. Claims are "
            "limited to diagnostic active-learning quality, label coverage, stop-policy, and runtime behavior for "
            "the configured synthetic or explicitly capped real-data fixtures; this is not native external-library "
            "workflow evidence and not proof of broad production superiority. Stochastic and committee proxy rows "
            "are integration/diagnostic evidence for SDK strategy wiring only, not true MC-dropout or independently "
            "trained committee quality evidence."
        ),
        "strategy_evidence_notes": {
            "stochastic_committee_proxy": (
                "The benchmark adapter creates deterministic perturbations from one sklearn model for stochastic "
                "and committee-style strategies. These rows validate integration and diagnostics only; they are not "
                "evidence of true MC-dropout or independently trained committee acquisition quality."
            )
        },
        "datasets": datasets,
        "strategies": strategies,
        "budgets": budgets,
        "skipped_budgets": budget_warning_rows,
        "seeds": seeds,
        "seed_count": len(set(seeds)),
        "initial_seed_size": args.initial_seed_size,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "allow_uncapped_real_standard": args.allow_uncapped_real_standard,
        "real_evidence_level": real_evidence_level(
            args.preset,
            len(set(seeds)),
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            allow_uncapped_real_standard=args.allow_uncapped_real_standard,
        ),
        "real_standard_contract": {
            "standard_presets": sorted(REAL_STANDARD_PRESETS),
            "required_seed_count": REAL_STANDARD_MIN_SEEDS,
            "requires_positive_max_train_samples": True,
            "requires_positive_max_test_samples": True,
            "required_calibration_columns": list(CALIBRATION_COLUMNS),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "stop_policies": [stop_policy_spec_to_dict(policy) for policy in stop_policy_specs],
        "artifacts": artifact_names,
        "artifact_names": sorted(set(artifact_names.values())),
        "sdk_usage": "Uses active_learning_sdk.engine.StrategyScheduler and SelectionContext for acquisition.",
        "sdk_gap": (
            "No current SDK gap for public initial-label import: ActiveLearningProject.import_labels(...) "
            "imports seed and validation labels through the public facade. This scheduler-level benchmark "
            "keeps seed labels in benchmark-owned state so strategy curves can be compared without project "
            "state-machine overhead."
        ),
    }

    write_csv(output_dir / "metrics.csv", metrics_rows)
    write_csv(output_dir / "selections.csv", selection_rows)
    write_csv(output_dir / "stop_policies.csv", stop_policy_rows)
    write_csv(output_dir / "full_train_reference.csv", full_reference_rows)
    write_csv(output_dir / "budget_warnings.csv", budget_warning_rows)
    write_strict_json(output_dir / "manifest.json", manifest)
    write_strict_json(
        output_dir / "validation.json",
        {
            "run_id": run_id,
            "datasets_checked": len(validation_rows),
            "checks": validation_rows,
        },
    )
    write_summary(output_dir, metrics_rows, selection_rows, stop_policy_rows, manifest, full_reference_rows)

    print(f"Wrote benchmark artifacts to {output_dir.resolve()}")
    print(f"Metrics rows: {len(metrics_rows)}")
    print(f"Selection rows: {len(selection_rows)}")
    print(f"Stop policy rows: {len(stop_policy_rows)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileExistsError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
