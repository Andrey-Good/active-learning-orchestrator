from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.reference_strategy_benchmark import (
    FORMULA_EQUIVALENT_STRATEGIES,
    add_curve_metrics,
    build_equivalence_rows,
    entropy_scores,
    least_confidence_scores,
    manual_entropy_select,
    manual_least_confidence_select,
    manual_margin_select,
    margin_scores,
    main,
    normalize_probability_rows,
    reference_strategies,
    run_one_curve,
    sdk_benchmark,
    sdk_reference_strategies,
    validate_probability_rows,
)


class FakeContext:
    def __init__(self, probabilities: dict[str, list[float]]) -> None:
        self.probabilities = probabilities

    def model_id(self) -> str:
        return "fixed-model"

    def predict_proba(self, sample_ids: list[str]) -> list[list[float]]:
        return [self.probabilities[sample_id] for sample_id in sample_ids]


def test_manual_score_formulas_match_expected_probability_ordering() -> None:
    probabilities = [
        [0.34, 0.33, 0.33],
        [0.60, 0.20, 0.20],
        [0.90, 0.05, 0.05],
    ]

    entropy = entropy_scores(probabilities)
    least_confidence = least_confidence_scores(probabilities)
    margin = margin_scores(probabilities)

    assert entropy[0] > entropy[1] > entropy[2]
    assert least_confidence[0] > least_confidence[1] > least_confidence[2]
    assert margin[0] > margin[1] > margin[2]


def test_manual_selectors_use_same_probability_rows_with_deterministic_ordering() -> None:
    context = FakeContext(
        {
            "most_uncertain": [0.34, 0.33, 0.33],
            "middle": [0.60, 0.20, 0.20],
            "confident": [0.90, 0.05, 0.05],
        }
    )
    pool_ids = ["confident", "middle", "most_uncertain"]

    assert manual_entropy_select(pool_ids, 3, context, seed=13)[0] == "most_uncertain"
    assert manual_least_confidence_select(pool_ids, 3, context, seed=13)[0] == "most_uncertain"
    assert manual_margin_select(pool_ids, 3, context, seed=13)[0] == "most_uncertain"


def test_probability_rows_validate_valid_rows_without_normalizing() -> None:
    rows = validate_probability_rows([[0.5, 0.5], [0.0, 1.0]], ["a", "b"])

    assert rows == [[0.5, 0.5], [0.0, 1.0]]
    assert normalize_probability_rows([[0.25, 0.75]], ["c"]) == [[0.25, 0.75]]


def test_probability_rows_reject_count_like_rows_instead_of_normalizing() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_probability_rows([[5.0, 5.0], [0.0, 1.0]], ["a", "b"])


def test_probability_rows_reject_logit_like_rows() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        validate_probability_rows([[2.0, -1.0]], ["a"])


def test_manual_selectors_reject_invalid_probability_rows() -> None:
    context = FakeContext({"count_like": [5.0, 5.0]})

    with pytest.raises(ValueError, match="sum to 1.0"):
        manual_entropy_select(["count_like"], 1, context, seed=13)


def test_manual_score_formulas_reject_invalid_probability_rows() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        entropy_scores([[5.0, 5.0]])
    with pytest.raises(ValueError, match="sum to 1.0"):
        margin_scores([[1.5, 0.5]])
    with pytest.raises(ValueError, match="non-negative"):
        least_confidence_scores([[2.0, -1.0]])


def test_random_is_not_part_of_formula_equivalence() -> None:
    assert FORMULA_EQUIVALENT_STRATEGIES == {
        "entropy": "manual_entropy",
        "margin": "manual_margin",
        "least_confidence": "manual_least_confidence",
        "class_group_balanced_entropy": "manual_class_group_balanced_entropy",
    }
    assert "random" not in FORMULA_EQUIVALENT_STRATEGIES


def test_external_library_formula_shims_are_labeled_honestly() -> None:
    strategies = reference_strategies(include_external=True)

    for name in [
        "modal_formula_entropy",
        "modal_formula_margin",
        "modal_formula_uncertainty",
        "skactiveml_formula_entropy",
        "skactiveml_formula_margin",
        "skactiveml_formula_least_confidence",
    ]:
        assert strategies[name].family == "formula_shim"
        assert strategies[name].skip_reason is None

    assert "modal_entropy" not in strategies
    assert "skactiveml_entropy" not in strategies


def test_equivalence_rows_only_compare_formula_equivalent_strategies() -> None:
    curve_selections = {
        ("dataset", "random", 13): {12: ["sdk-random"]},
        ("dataset", "manual_random", 13): {12: ["manual-random"]},
        ("dataset", "entropy", 13): {12: ["same"]},
        ("dataset", "manual_entropy", 13): {12: ["same"]},
    }

    rows = build_equivalence_rows(curve_selections)

    assert len(rows) == 1
    assert rows[0]["sdk_strategy"] == "entropy"
    assert rows[0]["reference_strategy"] == "manual_entropy"


def test_reference_curve_rows_satisfy_shared_curve_metric_contract() -> None:
    sdk_benchmark.ensure_benchmark_dependencies()
    dataset = sdk_benchmark.build_benchmark_dataset("separable_topics", seed=13)
    strategy = sdk_reference_strategies()["entropy"]

    metrics_rows, _selection_rows, _curve_selections = run_one_curve(
        dataset,
        strategy,
        budgets=[12],
        seed=13,
        initial_seed_size=9,
    )
    add_curve_metrics(metrics_rows)

    row = metrics_rows[0]
    for key in [
        "balanced_accuracy",
        "macro_recall",
        "zero_recall_class_count",
        "label_coverage_fraction",
        "missing_test_support_weighted_fraction",
        "aulc_balanced_accuracy",
        "lift_macro_recall_vs_random",
    ]:
        assert key in row


def test_reference_benchmark_refuses_non_empty_output_dir_without_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "reference"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("do not replace", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        main(
            [
                "--preset",
                "smoke",
                "--datasets",
                "separable_topics",
                "--budgets",
                "12",
                "--seeds",
                "13",
                "--output-dir",
                str(output_dir),
                "--no-include-external",
            ]
        )

    assert sentinel.read_text(encoding="utf-8") == "do not replace"


def test_reference_benchmark_cleans_non_empty_output_dir_with_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "reference"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("intentional local file", encoding="utf-8")

    assert (
        main(
            [
                "--preset",
                "smoke",
                "--datasets",
                "separable_topics",
                "--strategies",
                "entropy",
                "--budgets",
                "12",
                "--seeds",
                "13",
                "--output-dir",
                str(output_dir),
                "--no-include-external",
                "--overwrite",
            ]
        )
        == 0
    )

    assert (output_dir / "manifest.json").exists()
    assert not sentinel.exists()


def test_reference_benchmark_manifest_records_reproducibility_metadata(tmp_path: Path) -> None:
    output_dir = tmp_path / "reference"
    argv = [
        "--preset",
        "smoke",
        "--datasets",
        "separable_topics",
        "--strategies",
        "entropy,manual_entropy",
        "--budgets",
        "12",
        "--seeds",
        "13",
        "--output-dir",
        str(output_dir),
        "--no-include-external",
    ]

    assert main(argv) == 0

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_schema_version"] == sdk_benchmark.BENCHMARK_ARTIFACT_SCHEMA_VERSION
    assert manifest["argv"] == argv
    assert "sha" in manifest["git"]
    assert "status_entry_count" in manifest["git"]
    assert manifest["runtime"]["python_version_info"]["major"] >= 3
    assert manifest["artifacts"]["equivalence_csv"] == "equivalence.csv"
    assert manifest["artifacts"]["external_adapters_json"] == "external_adapters.json"
    assert "rows are not renormalized" in manifest["benchmark_contract"]
