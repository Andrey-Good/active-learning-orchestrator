from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import sdk_first_benchmark


EMBEDDING_STRATEGIES = {
    "coreset_kcenter",
    "embedding_kmeans_pp",
    "max_min_embedding",
    "deduplicate_near_neighbors",
    "density_weighted_diversity",
}

STAGE4_STRATEGIES = {
    "mc_dropout_entropy",
    "bald",
    "variation_ratio",
    "prediction_variance",
    "committee_vote_entropy",
    "committee_kl_divergence",
    "committee_pairwise_disagreement",
    "committee_margin",
}

HYBRID_STRATEGIES = {
    "hybrid_weighted_entropy_coreset",
    "hybrid_uncertainty_prefilter_coreset",
    "hybrid_diversity_prefilter_entropy",
    "hybrid_weighted_guarded",
}


def test_strategy_specs_include_adaptive_strategy() -> None:
    sdk_first_benchmark.ensure_benchmark_dependencies()

    specs = sdk_first_benchmark.strategy_specs()

    assert "adaptive_uncertainty_diversity" in specs
    assert specs["adaptive_uncertainty_diversity"].scheduler_config.strategy == "adaptive_uncertainty_diversity"


def test_strategy_specs_include_embedding_strategies() -> None:
    sdk_first_benchmark.ensure_benchmark_dependencies()

    specs = sdk_first_benchmark.strategy_specs()

    assert EMBEDDING_STRATEGIES.issubset(specs)
    for strategy_name in EMBEDDING_STRATEGIES:
        assert specs[strategy_name].scheduler_config.strategy == strategy_name


def test_strategy_specs_include_badge() -> None:
    sdk_first_benchmark.ensure_benchmark_dependencies()

    specs = sdk_first_benchmark.strategy_specs()

    assert "badge" in specs
    assert specs["badge"].scheduler_config.strategy == "badge"


def test_strategy_specs_include_stage4_stochastic_and_committee_strategies() -> None:
    sdk_first_benchmark.ensure_benchmark_dependencies()

    specs = sdk_first_benchmark.strategy_specs()

    assert STAGE4_STRATEGIES.issubset(specs)
    for strategy_name in STAGE4_STRATEGIES:
        assert specs[strategy_name].scheduler_config.strategy == strategy_name


def test_strategy_specs_include_hybrid_benchmark_strategies() -> None:
    sdk_first_benchmark.ensure_benchmark_dependencies()

    specs = sdk_first_benchmark.strategy_specs()

    assert HYBRID_STRATEGIES.issubset(specs)
    for strategy_name in HYBRID_STRATEGIES:
        assert specs[strategy_name].scheduler_config.mode == "hybrid"
        assert specs[strategy_name].scheduler_config.hybrid is not None


def test_selection_diagnostics_emit_duplicate_rate_and_redundancy_metrics() -> None:
    dataset = sdk_first_benchmark.make_grouped_duplicates(seed=13)
    train_ids = sdk_first_benchmark._sample_ids_by_split(dataset, "train")
    selected_ids = [train_ids[0], train_ids[0], train_ids[1]]
    labeled_ids = list(selected_ids)

    diagnostics = sdk_first_benchmark.compute_selection_diagnostics(
        selected_ids,
        labeled_ids,
        dataset,
        selected_embeddings=[
            [0.0, 0.0],
            [0.0, 0.0],
            [3.0, 4.0],
        ],
    )

    assert diagnostics["duplicate_selected_count"] == 1
    assert math.isclose(diagnostics["selected_duplicate_rate"], 1 / 3)
    assert math.isclose(diagnostics["selected_nn_distance_mean"], 5 / 3)
    assert diagnostics["selected_nn_distance_min"] == 0.0
    assert "group_hhi" in diagnostics
    assert "top_group_fraction" in diagnostics
    json.dumps(sdk_first_benchmark.sanitize_json_value(diagnostics), allow_nan=False)


def test_capped_real_dataset_labels_are_computed_from_retained_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSplit(list):
        features: dict[str, object] = {}

    fake_dataset = {
        "train": FakeSplit(
            [
                {"text": "discarded train example", "label": 0, "label_text": "discarded_train_label"},
            ]
        ),
        "test": FakeSplit(
            [
                {"text": "retained test example", "label": 1, "label_text": "retained_test_label"},
                {"text": "discarded test example", "label": 2, "label_text": "discarded_test_label"},
            ]
        ),
    }
    monkeypatch.setattr(sdk_first_benchmark, "load_hf_dataset", lambda path, config: fake_dataset)

    dataset = sdk_first_benchmark.make_real_dataset(
        "banking77",
        seed=13,
        max_train_samples=0,
        max_test_samples=1,
    )

    retained_labels = sorted({sample.label for sample in dataset.samples})
    assert dataset.labels == retained_labels == ["retained_test_label"]
    assert all(sample.label != "discarded_train_label" for sample in dataset.samples)
    assert all(sample.label != "discarded_test_label" for sample in dataset.samples)


def test_run_one_curve_embeds_selected_samples_before_retraining(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk_first_benchmark.ensure_benchmark_dependencies()

    events: list[tuple[str, int, tuple[str, ...]]] = []

    class FakeModel:
        def __init__(self, labels: list[str], seed: int) -> None:
            self.labels = list(labels)
            self.seed = seed
            self.fit_count = 0

        def fit(self, texts: list[str], labels: list[str], **_: object) -> None:
            self.fit_count += 1
            events.append(("fit", self.fit_count, tuple(texts)))

        def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
            del batch_size
            events.append(("embed", self.fit_count, tuple(texts)))
            if self.fit_count == 1:
                return [[0.0], [10.0]][: len(texts)]
            return [[0.0], [1.0]][: len(texts)]

    class FakeScheduler:
        def __init__(self, config: object) -> None:
            self.config = config

        def select_batch(
            self,
            pool_ids: list[str],
            batch_size: int,
            context: object,
            state: dict[str, object],
        ) -> tuple[list[str], dict[str, object]]:
            del context, state
            return pool_ids[:batch_size], {"mode": "fake"}

    def fake_train_and_evaluate(
        model: FakeModel,
        dataset: sdk_first_benchmark.BenchmarkDataset,
        labeled_ids: list[str],
        test_ids: list[str],
    ) -> dict[str, float]:
        sample_by_id = {sample.sample_id: sample for sample in dataset.samples}
        model.fit(
            [sample_by_id[sample_id].text for sample_id in labeled_ids],
            [sample_by_id[sample_id].label for sample_id in labeled_ids],
        )
        return {
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "macro_f1": 1.0,
            "weighted_f1": 1.0,
            "macro_recall": 1.0,
            "rare_recall": float("nan"),
            "zero_recall_class_count": 0.0,
            "zero_recall_class_fraction": 0.0,
        }

    dataset = sdk_first_benchmark.BenchmarkDataset(
        name="unit_curve",
        description="unit test curve",
        labels=["a", "b"],
        samples=[
            sdk_first_benchmark.BenchmarkSample("train_a0", "seed a", "a", "g0", "train"),
            sdk_first_benchmark.BenchmarkSample("train_b0", "seed b", "b", "g1", "train"),
            sdk_first_benchmark.BenchmarkSample("train_a1", "selected a", "a", "g2", "train"),
            sdk_first_benchmark.BenchmarkSample("train_b1", "selected b", "b", "g3", "train"),
            sdk_first_benchmark.BenchmarkSample("test_a0", "test a", "a", "g4", "test"),
            sdk_first_benchmark.BenchmarkSample("test_b0", "test b", "b", "g5", "test"),
        ],
    )

    monkeypatch.setattr(sdk_first_benchmark, "SklearnTextBenchmarkAdapter", FakeModel)
    monkeypatch.setattr(sdk_first_benchmark, "StrategyScheduler", FakeScheduler)
    monkeypatch.setattr(sdk_first_benchmark, "train_and_evaluate", fake_train_and_evaluate)

    _, selection_rows = sdk_first_benchmark.run_one_curve(
        dataset,
        sdk_first_benchmark.StrategySpec(
            name="fake_strategy",
            scheduler_config=sdk_first_benchmark.SchedulerConfig(mode="single", strategy="random"),
        ),
        budgets=[4],
        seed=13,
        initial_seed_size=2,
    )

    assert selection_rows[0]["selected_nn_distance_min"] == 10.0
    assert selection_rows[0]["selected_nn_distance_mean"] == 10.0
    event_names = [event[0] for event in events]
    assert event_names == ["fit", "embed", "fit"]
    assert events[1][1] == 1


def test_stop_policy_plateau_triggers_after_patience_and_min_budget() -> None:
    metrics_rows = [
        _stop_metric_row(budget=12, macro_f1=0.50, runtime_seconds=1.0),
        _stop_metric_row(budget=18, macro_f1=0.505, runtime_seconds=2.0),
        _stop_metric_row(budget=24, macro_f1=0.510, runtime_seconds=3.0),
    ]
    policy = sdk_first_benchmark.StopPolicySpec(
        name="test_plateau",
        policy_type="plateau",
        metric_name="macro_f1",
        min_budget=18,
        patience=1,
        min_delta=0.01,
    )

    rows = sdk_first_benchmark.simulate_stop_policies(metrics_rows, [policy])

    assert len(rows) == 1
    row = rows[0]
    assert row["stop_reason"] == "macro_f1_plateau_patience_1"
    assert row["stopped_budget"] == 18
    assert row["full_budget"] == 24
    assert row["labels_saved"] == 6
    assert math.isclose(row["relative_savings"], 0.25)
    assert math.isclose(row["runtime_at_stop"], 3.0)
    assert math.isclose(row["runtime_full"], 6.0)
    assert math.isclose(row["runtime_saved"], 3.0)


def test_stop_policy_no_stop_falls_back_to_final_budget_with_zero_savings() -> None:
    metrics_rows = [
        _stop_metric_row(budget=12, macro_f1=0.50, runtime_seconds=1.0),
        _stop_metric_row(budget=18, macro_f1=0.60, runtime_seconds=2.0),
        _stop_metric_row(budget=24, macro_f1=0.70, runtime_seconds=3.0),
    ]
    policy = sdk_first_benchmark.StopPolicySpec(
        name="test_no_stop",
        policy_type="plateau",
        metric_name="macro_f1",
        min_budget=12,
        patience=1,
        min_delta=0.001,
    )

    rows = sdk_first_benchmark.simulate_stop_policies(metrics_rows, [policy])

    assert len(rows) == 1
    row = rows[0]
    assert row["stop_reason"] == "final_budget_no_stop"
    assert row["stopped_budget"] == 24
    assert row["full_budget"] == 24
    assert row["labels_saved"] == 0
    assert row["relative_savings"] == 0.0
    assert row["quality_delta_vs_full"] == 0.0
    assert row["runtime_saved"] == 0.0


def test_stop_policy_rows_are_strict_json_serializable_with_nan_metrics() -> None:
    metrics_rows = [
        _stop_metric_row(budget=12, rare_recall=float("nan"), runtime_seconds=1.0),
        _stop_metric_row(budget=18, rare_recall=float("nan"), runtime_seconds=2.0),
    ]
    policy = sdk_first_benchmark.StopPolicySpec(
        name="rare_recall_plateau",
        policy_type="plateau",
        metric_name="rare_recall",
        min_budget=12,
        patience=1,
        min_delta=0.001,
    )

    rows = sdk_first_benchmark.simulate_stop_policies(metrics_rows, [policy])

    assert rows[0]["stop_metric"] is None
    assert rows[0]["full_metric"] is None
    assert rows[0]["quality_delta_vs_full"] is None
    json.dumps(rows, allow_nan=False)


def test_tiny_benchmark_run_writes_stop_policy_artifacts(tmp_path: Path) -> None:
    argv = [
        "--preset",
        "smoke",
        "--datasets",
        "grouped_duplicates",
        "--strategies",
        "random",
        "--budgets",
        "12,18",
        "--seeds",
        "13",
        "--output-dir",
        str(tmp_path),
    ]
    sdk_first_benchmark.main(argv)

    stop_policy_path = tmp_path / "stop_policies.csv"
    assert stop_policy_path.exists()
    with stop_policy_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == len(sdk_first_benchmark.default_stop_policies())
    assert {"policy_name", "stop_reason", "labels_saved", "quality_delta_vs_full"}.issubset(rows[0])

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert summary["row_counts"]["stop_policies"] == len(rows)
    assert len(summary["stop_policy_rows"]) == len(rows)
    assert manifest["artifacts"]["stop_policies_csv"] == "stop_policies.csv"
    assert "stop_policies" in manifest
    assert manifest["artifact_schema_version"] == sdk_first_benchmark.BENCHMARK_ARTIFACT_SCHEMA_VERSION
    assert manifest["argv"] == argv
    assert "sha" in manifest["git"]
    assert isinstance(manifest["git"]["dirty"], bool)
    assert manifest["runtime"]["python_version_info"]["major"] == sys.version_info.major
    assert manifest["runtime"]["platform"]


def test_sdk_first_benchmark_refuses_non_empty_output_dir_without_overwrite(tmp_path: Path) -> None:
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("do not replace", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        sdk_first_benchmark.main(
            [
                "--preset",
                "smoke",
                "--datasets",
                "grouped_duplicates",
                "--strategies",
                "random",
                "--budgets",
                "12",
                "--seeds",
                "13",
                "--output-dir",
                str(tmp_path),
            ]
        )

    assert sentinel.read_text(encoding="utf-8") == "do not replace"


def test_tiny_embedding_strategy_curve_completes_and_records_redundancy_metrics() -> None:
    sdk_first_benchmark.ensure_benchmark_dependencies()
    dataset = sdk_first_benchmark.make_grouped_duplicates(seed=13)
    strategy = sdk_first_benchmark.strategy_specs()["coreset_kcenter"]

    metrics_rows, selection_rows = sdk_first_benchmark.run_one_curve(
        dataset,
        strategy,
        budgets=[12],
        seed=13,
        initial_seed_size=9,
    )

    assert len(metrics_rows) == 1
    assert len(selection_rows) == 1
    selection = selection_rows[0]
    assert selection["strategy"] == "coreset_kcenter"
    assert selection["selected_duplicate_rate"] == 0.0
    assert selection["selected_nn_distance_mean"] is not None
    assert selection["selected_nn_distance_min"] is not None
    assert float(selection["selected_nn_distance_mean"]) >= float(selection["selected_nn_distance_min"])
    json.dumps(sdk_first_benchmark.sanitize_json_value(selection), allow_nan=False)


def test_tiny_badge_curve_completes_and_records_runtime_and_redundancy_metrics() -> None:
    sdk_first_benchmark.ensure_benchmark_dependencies()
    dataset = sdk_first_benchmark.make_grouped_duplicates(seed=13)
    strategy = sdk_first_benchmark.strategy_specs()["badge"]

    metrics_rows, selection_rows = sdk_first_benchmark.run_one_curve(
        dataset,
        strategy,
        budgets=[12],
        seed=13,
        initial_seed_size=9,
    )

    assert len(metrics_rows) == 1
    assert len(selection_rows) == 1
    metric = metrics_rows[0]
    selection = selection_rows[0]
    assert metric["strategy"] == "badge"
    assert metric["runtime_seconds"] >= 0.0
    assert metric["selected_count"] == 3
    assert selection["strategy"] == "badge"
    assert math.isfinite(selection["runtime_seconds"])
    assert selection["runtime_seconds"] >= 0.0
    assert selection["selected_duplicate_rate"] == 0.0
    assert selection["selected_nn_distance_mean"] is not None
    assert selection["selected_nn_distance_min"] is not None
    assert float(selection["selected_nn_distance_mean"]) >= float(selection["selected_nn_distance_min"])
    json.dumps(sdk_first_benchmark.sanitize_json_value(metric), allow_nan=False)
    json.dumps(sdk_first_benchmark.sanitize_json_value(selection), allow_nan=False)


def test_sklearn_badge_gradient_proxy_is_bounded_and_deterministic() -> None:
    sdk_first_benchmark.ensure_benchmark_dependencies()
    adapter = sdk_first_benchmark.SklearnTextBenchmarkAdapter(labels=["a", "b", "c"], seed=13)
    adapter.fit(
        [
            "alpha bright token",
            "beta dark token",
            "gamma calm token",
            "alpha extra phrase",
            "beta extra phrase",
            "gamma extra phrase",
        ],
        ["a", "b", "c", "a", "b", "c"],
    )

    first = adapter.gradient_embed(["alpha bright token", "gamma calm token"])
    second = adapter.gradient_embed(["alpha bright token", "gamma calm token"])

    assert first == second
    assert len(first) == 2
    assert {len(row) for row in first} == {adapter.BADGE_GRADIENT_EMBEDDING_DIM}
    assert any(value != 0.0 for row in first for value in row)


def _stop_metric_row(
    *,
    budget: int,
    macro_f1: float = 0.5,
    accuracy: float = 0.5,
    weighted_f1: float = 0.5,
    rare_recall: float | None = None,
    runtime_seconds: float = 0.0,
) -> dict[str, object]:
    return {
        "dataset": "unit_dataset",
        "strategy": "unit_strategy",
        "seed": 13,
        "budget": budget,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "rare_recall": rare_recall,
        "runtime_seconds": runtime_seconds,
    }


def test_tiny_stochastic_strategy_curve_completes_and_records_runtime_and_selection_diagnostics() -> None:
    sdk_first_benchmark.ensure_benchmark_dependencies()
    dataset = sdk_first_benchmark.make_grouped_duplicates(seed=13)
    strategy = sdk_first_benchmark.strategy_specs()["bald"]

    metrics_rows, selection_rows = sdk_first_benchmark.run_one_curve(
        dataset,
        strategy,
        budgets=[12],
        seed=13,
        initial_seed_size=9,
    )

    assert len(metrics_rows) == 1
    assert len(selection_rows) == 1
    metric = metrics_rows[0]
    selection = selection_rows[0]
    assert metric["strategy"] == "bald"
    assert metric["runtime_seconds"] >= 0.0
    assert metric["selected_count"] == 3
    assert selection["strategy"] == "bald"
    assert math.isfinite(selection["runtime_seconds"])
    assert selection["runtime_seconds"] >= 0.0
    assert selection["selected_duplicate_rate"] >= 0.0
    assert selection["selected_nn_distance_mean"] is not None
    assert selection["selected_nn_distance_min"] is not None
    assert "selected_label_counts" in selection
    assert "cumulative_label_counts" in selection
    assert "selected_group_counts" in selection
    json.dumps(sdk_first_benchmark.sanitize_json_value(metric), allow_nan=False)
    json.dumps(sdk_first_benchmark.sanitize_json_value(selection), allow_nan=False)


def test_tiny_committee_strategy_curve_completes_and_records_runtime_and_selection_diagnostics() -> None:
    sdk_first_benchmark.ensure_benchmark_dependencies()
    dataset = sdk_first_benchmark.make_grouped_duplicates(seed=13)
    strategy = sdk_first_benchmark.strategy_specs()["committee_vote_entropy"]

    metrics_rows, selection_rows = sdk_first_benchmark.run_one_curve(
        dataset,
        strategy,
        budgets=[12],
        seed=13,
        initial_seed_size=9,
    )

    assert len(metrics_rows) == 1
    assert len(selection_rows) == 1
    metric = metrics_rows[0]
    selection = selection_rows[0]
    assert metric["strategy"] == "committee_vote_entropy"
    assert metric["runtime_seconds"] >= 0.0
    assert metric["selected_count"] == 3
    assert selection["strategy"] == "committee_vote_entropy"
    assert math.isfinite(selection["runtime_seconds"])
    assert selection["runtime_seconds"] >= 0.0
    assert selection["selected_duplicate_rate"] >= 0.0
    assert selection["selected_nn_distance_mean"] is not None
    assert selection["selected_nn_distance_min"] is not None
    assert "selected_label_counts" in selection
    assert "cumulative_label_counts" in selection
    assert "selected_group_counts" in selection
    json.dumps(sdk_first_benchmark.sanitize_json_value(metric), allow_nan=False)
    json.dumps(sdk_first_benchmark.sanitize_json_value(selection), allow_nan=False)


def test_tiny_hybrid_weighted_curve_completes() -> None:
    sdk_first_benchmark.ensure_benchmark_dependencies()
    dataset = sdk_first_benchmark.make_grouped_duplicates(seed=13)
    strategy = sdk_first_benchmark.strategy_specs()["hybrid_weighted_entropy_coreset"]

    metrics_rows, selection_rows = sdk_first_benchmark.run_one_curve(
        dataset,
        strategy,
        budgets=[12],
        seed=13,
        initial_seed_size=9,
    )

    assert len(metrics_rows) == 1
    assert len(selection_rows) == 1
    metric = metrics_rows[0]
    selection = selection_rows[0]
    snapshot = json.loads(selection["scheduler_snapshot"])
    assert metric["strategy"] == "hybrid_weighted_entropy_coreset"
    assert metric["runtime_seconds"] >= 0.0
    assert metric["selected_count"] == 3
    assert selection["strategy"] == "hybrid_weighted_entropy_coreset"
    assert snapshot["mode"] == "hybrid"
    assert snapshot["hybrid"]["mode"] == "weighted"
    json.dumps(sdk_first_benchmark.sanitize_json_value(metric), allow_nan=False)
    json.dumps(sdk_first_benchmark.sanitize_json_value(selection), allow_nan=False)


def test_tiny_hybrid_guarded_curve_completes_and_records_diagnostics() -> None:
    sdk_first_benchmark.ensure_benchmark_dependencies()
    dataset = sdk_first_benchmark.make_grouped_duplicates(seed=13)
    strategy = sdk_first_benchmark.strategy_specs()["hybrid_weighted_guarded"]

    metrics_rows, selection_rows = sdk_first_benchmark.run_one_curve(
        dataset,
        strategy,
        budgets=[12],
        seed=13,
        initial_seed_size=9,
    )

    assert len(metrics_rows) == 1
    assert len(selection_rows) == 1
    metric = metrics_rows[0]
    selection = selection_rows[0]
    snapshot = json.loads(selection["scheduler_snapshot"])
    assert metric["strategy"] == "hybrid_weighted_guarded"
    assert metric["runtime_seconds"] >= 0.0
    assert metric["selected_count"] == 3
    assert selection["strategy"] == "hybrid_weighted_guarded"
    assert math.isfinite(selection["runtime_seconds"])
    assert selection["runtime_seconds"] >= 0.0
    assert selection["selected_duplicate_rate"] >= 0.0
    assert selection["selected_nn_distance_mean"] is not None
    assert selection["selected_nn_distance_min"] is not None
    assert "selected_group_counts" in selection
    assert "top_group_fraction" in selection
    assert "group_hhi" in selection
    assert snapshot["mode"] == "hybrid"
    assert snapshot["hybrid"]["class_balance"] is True
    assert snapshot["hybrid"]["group_balance"] is True
    json.dumps(sdk_first_benchmark.sanitize_json_value(metric), allow_nan=False)
    json.dumps(sdk_first_benchmark.sanitize_json_value(selection), allow_nan=False)
