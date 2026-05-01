from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import sdk_first_benchmark


class FakeClassLabel:
    def __init__(self, names: list[str]) -> None:
        self.names = names

    def int2str(self, value: int) -> str:
        return self.names[value]


class FakeSplit(list[dict[str, object]]):
    def __init__(self, rows: list[dict[str, object]], label_column: str, label_names: list[str]) -> None:
        super().__init__(rows)
        self.features = {label_column: FakeClassLabel(label_names)}


def fake_hf_dataset() -> dict[str, FakeSplit]:
    labels = ["billing", "card", "loan"]
    return {
        "train": FakeSplit(
            [
                {"text": "billing question", "label": 0},
                {"text": "card blocked", "label": 1},
                {"text": "loan rate", "label": 2},
                {"text": "billing invoice", "label": 0},
            ],
            "label",
            labels,
        ),
        "test": FakeSplit(
            [
                {"text": "billing support", "label": 0},
                {"text": "card support", "label": 1},
                {"text": "loan support", "label": 2},
            ],
            "label",
            labels,
        ),
    }


def test_real_dataset_registry_contains_expected_metadata() -> None:
    registry = sdk_first_benchmark.REAL_DATASET_REGISTRY

    assert registry["clinc_oos_imbalanced"].hf_path == "clinc/clinc_oos"
    assert registry["clinc_oos_imbalanced"].hf_config == "imbalanced"
    assert registry["clinc_oos_imbalanced"].splits == ("train", "validation", "test")
    assert registry["clinc_oos_imbalanced"].label_column == "intent"
    assert registry["clinc_oos_plus"].hf_config == "plus"
    assert registry["banking77"].hf_path == "mteb/banking77"
    assert registry["banking77"].splits == ("train", "test")
    assert registry["banking77"].label_name_column == "label_text"
    assert registry["dair_ai_emotion"].hf_path == "dair-ai/emotion"
    assert registry["dair_ai_emotion"].sanity_easy_coverage is True


def test_real_dataset_loader_can_be_faked_without_hf_download(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_loader(path: str, config: str) -> dict[str, FakeSplit]:
        calls.append((path, config))
        return fake_hf_dataset()

    monkeypatch.setattr(sdk_first_benchmark, "load_hf_dataset", fake_loader)

    dataset = sdk_first_benchmark.make_real_dataset("banking77", seed=13)

    assert calls == [("mteb/banking77", "default")]
    assert dataset.labels == ["billing", "card", "loan"]
    assert {sample.split for sample in dataset.samples} == {"train", "test"}
    assert all(sample.sample_id.startswith("ds103_") for sample in dataset.samples)
    assert all(sample.group_id.startswith("ds103_") for sample in dataset.samples)


def test_real_dataset_loader_prefers_explicit_label_name_column(monkeypatch) -> None:
    labels = ["numeric_zero", "numeric_one"]
    fake_dataset = {
        "train": FakeSplit(
            [
                {"text": "zero text", "label": 0, "label_text": labels[0]},
                {"text": "one text", "label": 1, "label_text": labels[1]},
            ],
            "label",
            ["0", "1"],
        ),
        "test": FakeSplit(
            [
                {"text": "zero test", "label": 0, "label_text": labels[0]},
                {"text": "one test", "label": 1, "label_text": labels[1]},
            ],
            "label",
            ["0", "1"],
        ),
    }
    monkeypatch.setattr(sdk_first_benchmark, "load_hf_dataset", lambda path, config: fake_dataset)

    dataset = sdk_first_benchmark.make_real_dataset("banking77", seed=13)

    assert dataset.labels == sorted(labels)


def test_real_dataset_loader_applies_deterministic_train_and_test_caps(monkeypatch) -> None:
    monkeypatch.setattr(sdk_first_benchmark, "load_hf_dataset", lambda path, config: fake_hf_dataset())

    first = sdk_first_benchmark.make_real_dataset("banking77", seed=13, max_train_samples=2, max_test_samples=1)
    second = sdk_first_benchmark.make_real_dataset("banking77", seed=13, max_train_samples=2, max_test_samples=1)

    assert [sample.sample_id for sample in first.samples] == [sample.sample_id for sample in second.samples]
    assert sum(1 for sample in first.samples if sample.split == "train") == 2
    assert sum(1 for sample in first.samples if sample.split == "test") == 1


def test_opaque_real_ids_and_groups_do_not_leak_labels_or_numeric_label_ids(monkeypatch) -> None:
    monkeypatch.setattr(sdk_first_benchmark, "load_hf_dataset", lambda path, config: fake_hf_dataset())

    dataset = sdk_first_benchmark.make_real_dataset("banking77", seed=13)
    visible_values = [value for sample in dataset.samples for value in (sample.sample_id, sample.group_id)]

    for value in visible_values:
        assert "billing" not in value
        assert "card" not in value
        assert "loan" not in value
        assert "_0_" not in value
        assert "_1_" not in value
        assert "_2_" not in value


def test_label_coverage_and_missing_label_metrics_are_correct() -> None:
    dataset = sdk_first_benchmark.BenchmarkDataset(
        name="toy",
        description="toy",
        labels=["a", "b", "c"],
        samples=[
            sdk_first_benchmark.BenchmarkSample("s1", "a train", "a", "g1", "train"),
            sdk_first_benchmark.BenchmarkSample("s2", "b train", "b", "g2", "train"),
            sdk_first_benchmark.BenchmarkSample("s3", "c train", "c", "g3", "train"),
            sdk_first_benchmark.BenchmarkSample("t1", "a test", "a", "g4", "test"),
            sdk_first_benchmark.BenchmarkSample("t2", "b test", "b", "g5", "test"),
            sdk_first_benchmark.BenchmarkSample("t3", "b test", "b", "g6", "test"),
            sdk_first_benchmark.BenchmarkSample("t4", "c test", "c", "g7", "test"),
        ],
    )

    metrics = sdk_first_benchmark.compute_label_coverage_metrics(dataset, ["s1", "s2"], ["s2"])

    assert metrics["label_coverage_count"] == 2
    assert metrics["label_coverage_fraction"] == 2 / 3
    assert metrics["missing_label_count"] == 1
    assert metrics["missing_label_fraction"] == 1 / 3
    assert metrics["missing_test_support_weighted_fraction"] == 1 / 4
    assert metrics["new_labels_selected_count"] == 1
    assert metrics["new_labels_selected_fraction"] == 1.0


def test_initial_seed_randomizes_label_subset_when_budget_is_below_class_count() -> None:
    labels = [f"label_{index}" for index in range(6)]
    samples = [
        sdk_first_benchmark.BenchmarkSample(f"s{index}", f"text {label}", label, f"g{index}", "train")
        for index, label in enumerate(labels)
    ]
    dataset = sdk_first_benchmark.BenchmarkDataset(
        name="toy_many_class",
        description="toy",
        labels=labels,
        samples=samples,
    )
    train_ids = [sample.sample_id for sample in samples]

    selected_label_sets = {
        tuple(
            sorted(
                next(sample.label for sample in samples if sample.sample_id == sample_id)
                for sample_id in sdk_first_benchmark.choose_initial_seed(dataset, train_ids, 3, seed)
            )
        )
        for seed in range(1, 8)
    }

    assert len(selected_label_sets) > 1
    assert selected_label_sets != {("label_0", "label_1", "label_2")}


def test_zero_recall_ignores_labels_without_test_support() -> None:
    sdk_first_benchmark.ensure_benchmark_dependencies()

    class PerfectPipeline:
        def predict(self, texts):
            return ["a", "b"]

    class PerfectModel:
        pipeline = PerfectPipeline()

        def fit(self, texts, labels):
            return None

        def evaluate(self, texts, labels):
            return {
                "accuracy": 1.0,
                "balanced_accuracy": 1.0,
                "macro_f1": 1.0,
                "weighted_f1": 1.0,
                "macro_recall": 1.0,
            }

        def predict_proba(self, texts):
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    dataset = sdk_first_benchmark.BenchmarkDataset(
        name="toy_missing_test_support",
        description="toy",
        labels=["a", "b", "c"],
        samples=[
            sdk_first_benchmark.BenchmarkSample("s1", "a train", "a", "g1", "train"),
            sdk_first_benchmark.BenchmarkSample("s2", "b train", "b", "g2", "train"),
            sdk_first_benchmark.BenchmarkSample("s3", "c train", "c", "g3", "train"),
            sdk_first_benchmark.BenchmarkSample("t1", "a test", "a", "g4", "test"),
            sdk_first_benchmark.BenchmarkSample("t2", "b test", "b", "g5", "test"),
        ],
    )

    metrics = sdk_first_benchmark.train_and_evaluate(PerfectModel(), dataset, ["s1", "s2", "s3"], ["t1", "t2"])

    assert metrics["zero_recall_class_count"] == 0.0
    assert metrics["zero_recall_class_fraction"] == 0.0
    assert metrics["multiclass_brier_score"] == 0.0
    assert metrics["nll"] == 0.0
    assert metrics["ece"] == 0.0


def test_calibration_metrics_are_correct_for_simple_fixture() -> None:
    metrics = sdk_first_benchmark.compute_calibration_metrics(
        ["a", "b", "b"],
        [[0.8, 0.2], [0.4, 0.6], [0.1, 0.9]],
        ["a", "b"],
    )

    assert math.isclose(metrics["multiclass_brier_score"], 0.14)
    assert math.isclose(metrics["nll"], -(math.log(0.8) + math.log(0.6) + math.log(0.9)) / 3)
    assert math.isclose(metrics["ece"], ((1.0 - 0.8) + (1.0 - 0.6) + (1.0 - 0.9)) / 3)


def test_cli_parser_accepts_real_presets_and_dataset_names() -> None:
    args = sdk_first_benchmark.build_parser().parse_args(
        [
            "--preset",
            "real_smoke",
            "--datasets",
            "banking77,dair_ai_emotion",
            "--max-train-samples",
            "300",
            "--max-test-samples",
            "200",
        ]
    )

    assert args.preset == "real_smoke"
    assert args.max_train_samples == 300
    assert args.max_test_samples == 200
    assert sdk_first_benchmark.build_parser().parse_args(["--preset", "real_medium"]).preset == "real_medium"
    assert sdk_first_benchmark.build_parser().parse_args(["--preset", "real_full"]).preset == "real_full"
    assert sdk_first_benchmark.parse_csv_list(args.datasets) == ["banking77", "dair_ai_emotion"]
    assert sdk_first_benchmark.default_budgets_for_preset("real_smoke") == [100, 200, 300, 500, 800]


def test_real_standard_presets_reject_missing_caps_and_too_few_seeds(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="requires at least 3 distinct seeds"):
        sdk_first_benchmark.main(
            [
                "--preset",
                "real_medium",
                "--seeds",
                "13,21",
                "--max-train-samples",
                "300",
                "--max-test-samples",
                "200",
                "--output-dir",
                str(tmp_path / "too_few_seeds"),
            ]
        )

    with pytest.raises(SystemExit, match="requires an explicit positive --max-train-samples"):
        sdk_first_benchmark.main(
            [
                "--preset",
                "real_full",
                "--seeds",
                "13,21,34",
                "--max-test-samples",
                "200",
                "--output-dir",
                str(tmp_path / "missing_train_cap"),
            ]
        )

    with pytest.raises(SystemExit, match="requires an explicit positive --max-test-samples"):
        sdk_first_benchmark.main(
            [
                "--preset",
                "real_medium",
                "--seeds",
                "13,21,34",
                "--max-train-samples",
                "300",
                "--output-dir",
                str(tmp_path / "missing_test_cap"),
            ]
        )


def test_real_datasets_are_not_in_synthetic_defaults() -> None:
    smoke_datasets = set(sdk_first_benchmark.default_datasets_for_preset("smoke"))
    full_datasets = set(sdk_first_benchmark.default_datasets_for_preset("full"))
    real_datasets = set(sdk_first_benchmark.REAL_DATASET_NAMES)

    assert smoke_datasets.isdisjoint(real_datasets)
    assert full_datasets.isdisjoint(real_datasets)
    assert sdk_first_benchmark.default_datasets_for_preset("real_smoke") == ["banking77"]
    assert sdk_first_benchmark.default_datasets_for_preset("real_medium") == [
        "banking77",
        "clinc_oos_imbalanced",
    ]
    assert sdk_first_benchmark.default_datasets_for_preset("real_full") == list(sdk_first_benchmark.REAL_DATASET_NAMES)


def test_real_many_class_initial_seed_allows_less_than_label_count(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sdk_first_benchmark, "load_hf_dataset", lambda path, config: fake_hf_dataset())

    sdk_first_benchmark.main(
        [
            "--preset",
            "real_smoke",
            "--datasets",
            "banking77",
            "--strategies",
            "random",
            "--budgets",
            "2",
            "--initial-seed-size",
            "2",
            "--output-dir",
            str(tmp_path),
        ]
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    metrics_header = (tmp_path / "metrics.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    reference_header = (tmp_path / "full_train_reference.csv").read_text(encoding="utf-8").splitlines()[0].split(",")

    assert summary["row_counts"]["metrics"] == 1
    assert summary["row_counts"]["full_train_reference"] == 1
    assert summary["manifest"]["real_evidence_level"] == "smoke_only"
    assert summary["manifest"]["seed_count"] == 1
    assert "multiclass_brier_score" in metrics_header
    assert "nll" in metrics_header
    assert "ece" in metrics_header
    assert "multiclass_brier_score" in reference_header
    assert "nll" in reference_header
    assert "ece" in reference_header
