from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.quality_gate_report import build_quality_gate_report, write_quality_gate_report


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_quality_gate_report_computes_aggregation_math(tmp_path: Path) -> None:
    rows = [
        {
            "dataset": "d1",
            "strategy": "random",
            "seed": 1,
            "budget": 10,
            "macro_f1": 0.20,
            "runtime_seconds": 1.0,
            "label_coverage_fraction": 0.50,
            "zero_recall_class_fraction": 0.40,
        },
        {
            "dataset": "d1",
            "strategy": "random",
            "seed": 1,
            "budget": 20,
            "macro_f1": 0.40,
            "runtime_seconds": 2.0,
            "label_coverage_fraction": 0.60,
            "zero_recall_class_fraction": 0.30,
        },
        {
            "dataset": "d1",
            "strategy": "entropy",
            "seed": 1,
            "budget": 10,
            "macro_f1": 0.30,
            "runtime_seconds": 1.5,
            "label_coverage_fraction": 0.55,
            "zero_recall_class_fraction": 0.35,
        },
        {
            "dataset": "d1",
            "strategy": "entropy",
            "seed": 1,
            "budget": 20,
            "macro_f1": 0.60,
            "runtime_seconds": 2.5,
            "label_coverage_fraction": 0.80,
            "zero_recall_class_fraction": 0.10,
        },
        {
            "dataset": "d1",
            "strategy": "random",
            "seed": 2,
            "budget": 10,
            "macro_f1": 0.50,
            "runtime_seconds": 1.1,
            "label_coverage_fraction": 0.70,
            "zero_recall_class_fraction": 0.20,
        },
        {
            "dataset": "d1",
            "strategy": "random",
            "seed": 2,
            "budget": 20,
            "macro_f1": 0.50,
            "runtime_seconds": 2.1,
            "label_coverage_fraction": 0.75,
            "zero_recall_class_fraction": 0.20,
        },
        {
            "dataset": "d1",
            "strategy": "entropy",
            "seed": 2,
            "budget": 10,
            "macro_f1": 0.45,
            "runtime_seconds": 1.6,
            "label_coverage_fraction": 0.72,
            "zero_recall_class_fraction": 0.20,
        },
        {
            "dataset": "d1",
            "strategy": "entropy",
            "seed": 2,
            "budget": 20,
            "macro_f1": 0.55,
            "runtime_seconds": 2.6,
            "label_coverage_fraction": 0.85,
            "zero_recall_class_fraction": 0.15,
        },
    ]
    for row in rows:
        row["multiclass_brier_score"] = 0.25
        row["nll"] = 0.60
        row["ece"] = 0.10
    _write_csv(tmp_path / "metrics.csv", rows)
    _write_csv(
        tmp_path / "full_train_reference.csv",
        [
            {
                "dataset": "d1",
                "seed": 1,
                "macro_f1": 0.80,
                "runtime_seconds": 10.0,
                "multiclass_brier_score": 0.20,
                "nll": 0.50,
                "ece": 0.08,
            },
            {
                "dataset": "d1",
                "seed": 2,
                "macro_f1": 1.00,
                "runtime_seconds": 10.0,
                "multiclass_brier_score": 0.18,
                "nll": 0.45,
                "ece": 0.07,
            },
        ],
    )

    report = write_quality_gate_report(tmp_path)

    entropy_budget_20 = next(
        row
        for row in report["mean_std_by_dataset_strategy_budget"]
        if row["dataset"] == "d1" and row["strategy"] == "entropy" and row["budget"] == 20
    )
    assert math.isclose(entropy_budget_20["macro_f1_mean"], 0.575)
    assert math.isclose(entropy_budget_20["runtime_seconds_mean"], 2.55)

    final_lift = next(row for row in report["final_budget_lift_vs_random"] if row["strategy"] == "entropy")
    assert math.isclose(final_lift["mean_lift"], 0.125)

    win_rate = next(row for row in report["strategy_win_rate_vs_random"] if row["strategy"] == "entropy")
    assert math.isclose(win_rate["win_rate"], 0.75)
    assert math.isclose(win_rate["non_loss_rate"], 0.75)
    assert win_rate["comparisons"] == 4

    seed_1_aulc = next(
        row
        for row in report["normalized_aulc"]
        if row["dataset"] == "d1" and row["strategy"] == "entropy" and row["seed"] == 1
    )
    assert math.isclose(seed_1_aulc["normalized_aulc"], 0.45)
    assert math.isclose(seed_1_aulc["normalized_aulc_vs_full_train"], 0.5625)

    aulc_lift = next(row for row in report["mean_aulc_lift_vs_random"] if row["strategy"] == "entropy")
    assert math.isclose(aulc_lift["mean_aulc_lift"], 0.075)

    coverage_delta = next(
        row
        for row in report["coverage_zero_recall_deltas_at_final_budget"]
        if row["dataset"] == "d1" and row["strategy"] == "entropy"
    )
    assert math.isclose(coverage_delta["label_coverage_fraction_delta_mean"], 0.15)
    assert math.isclose(coverage_delta["zero_recall_class_fraction_delta_mean"], -0.125)

    assert report["quality_gates"]["passed"] is True
    assert report["seed_summary"]["count"] == 2
    assert report["calibration_summary_at_final_budget"]
    calibration_row = next(
        row
        for row in report["calibration_summary_at_final_budget"]
        if row["dataset"] == "d1" and row["strategy"] == "entropy"
    )
    assert math.isclose(calibration_row["multiclass_brier_score_mean"], 0.25)
    assert (tmp_path / "quality_gate.json").exists()
    assert (tmp_path / "quality_gate.md").exists()
    json.loads((tmp_path / "quality_gate.json").read_text(encoding="utf-8"))
    assert "Calibration At Final Budget" in (tmp_path / "quality_gate.md").read_text(encoding="utf-8")


def test_quality_gate_report_fails_when_random_baseline_is_missing(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "metrics.csv",
        [
            {
                "dataset": "d1",
                "strategy": "entropy",
                "seed": 1,
                "budget": 10,
                "macro_f1": 0.30,
                "runtime_seconds": 1.5,
            },
            {
                "dataset": "d1",
                "strategy": "entropy",
                "seed": 1,
                "budget": 20,
                "macro_f1": 0.60,
                "runtime_seconds": 2.5,
            },
        ],
    )

    report = build_quality_gate_report(tmp_path)

    assert report["quality_gates"]["passed"] is False
    random_gate = next(check for check in report["quality_gates"]["checks"] if check["name"] == "random_baseline_present")
    assert random_gate["passed"] is False

    win_rate = next(row for row in report["strategy_win_rate_vs_random"] if row["strategy"] == "entropy")
    assert win_rate["win_rate"] is None
    assert win_rate["non_loss_rate"] is None
    assert win_rate["comparisons"] == 0
    assert win_rate["missing_baseline_rows"] == 2

    aulc_lift = next(row for row in report["mean_aulc_lift_vs_random"] if row["strategy"] == "entropy")
    assert aulc_lift["mean_aulc_lift"] is None
    assert aulc_lift["missing_baseline_rows"] == 1


def test_quality_gate_requires_one_strategy_to_pass_all_quality_checks(tmp_path: Path) -> None:
    rows = [
        {
            "dataset": "d1",
            "strategy": "random",
            "seed": 1,
            "budget": 10,
            "macro_f1": 0.50,
            "runtime_seconds": 1.0,
        },
        {
            "dataset": "d1",
            "strategy": "random",
            "seed": 1,
            "budget": 20,
            "macro_f1": 0.50,
            "runtime_seconds": 1.0,
        },
        {
            "dataset": "d1",
            "strategy": "final_only",
            "seed": 1,
            "budget": 10,
            "macro_f1": 0.10,
            "runtime_seconds": 1.0,
        },
        {
            "dataset": "d1",
            "strategy": "final_only",
            "seed": 1,
            "budget": 20,
            "macro_f1": 0.60,
            "runtime_seconds": 1.0,
        },
        {
            "dataset": "d1",
            "strategy": "aulc_only",
            "seed": 1,
            "budget": 10,
            "macro_f1": 0.90,
            "runtime_seconds": 1.0,
        },
        {
            "dataset": "d1",
            "strategy": "aulc_only",
            "seed": 1,
            "budget": 20,
            "macro_f1": 0.40,
            "runtime_seconds": 1.0,
        },
    ]
    _write_csv(tmp_path / "metrics.csv", rows)

    report = build_quality_gate_report(tmp_path)

    assert report["quality_gates"]["passed"] is False
    assert report["quality_gates"]["passing_strategies"] == []
    candidate_gate = next(
        check for check in report["quality_gates"]["checks"] if check["name"] == "single_strategy_quality_candidate"
    )
    assert candidate_gate["passed"] is False


def test_quality_gate_uses_meaningful_acquisition_non_loss_for_candidate(tmp_path: Path) -> None:
    rows = [
        {
            "dataset": "d1",
            "strategy": "random",
            "seed": 1,
            "budget": 10,
            "macro_f1": 0.90,
            "runtime_seconds": 1.0,
            "selected_count": 1,
        },
        {
            "dataset": "d1",
            "strategy": "random",
            "seed": 1,
            "budget": 20,
            "macro_f1": 0.40,
            "runtime_seconds": 1.0,
            "selected_count": 10,
        },
        {
            "dataset": "d1",
            "strategy": "random",
            "seed": 1,
            "budget": 40,
            "macro_f1": 0.50,
            "runtime_seconds": 1.0,
            "selected_count": 20,
        },
        {
            "dataset": "d1",
            "strategy": "candidate",
            "seed": 1,
            "budget": 10,
            "macro_f1": 0.10,
            "runtime_seconds": 1.0,
            "selected_count": 1,
        },
        {
            "dataset": "d1",
            "strategy": "candidate",
            "seed": 1,
            "budget": 20,
            "macro_f1": 0.60,
            "runtime_seconds": 1.0,
            "selected_count": 10,
        },
        {
            "dataset": "d1",
            "strategy": "candidate",
            "seed": 1,
            "budget": 40,
            "macro_f1": 0.70,
            "runtime_seconds": 1.0,
            "selected_count": 20,
        },
    ]
    _write_csv(tmp_path / "metrics.csv", rows)

    report = build_quality_gate_report(tmp_path)

    full_rates = next(row for row in report["strategy_win_rate_vs_random"] if row["strategy"] == "candidate")
    meaningful_rates = next(
        row
        for row in report["strategy_meaningful_acquisition_win_rate_vs_random"]
        if row["strategy"] == "candidate"
    )
    assert math.isclose(full_rates["non_loss_rate"], 2 / 3)
    assert meaningful_rates["comparisons"] == 2
    assert meaningful_rates["non_loss_rate"] == 1.0
    assert report["quality_gates"]["passing_strategies"] == ["candidate"]


def test_quality_gate_fails_when_strategy_only_ties_random(tmp_path: Path) -> None:
    rows = [
        {
            "dataset": "d1",
            "strategy": "random",
            "seed": 1,
            "budget": 10,
            "macro_f1": 0.80,
            "runtime_seconds": 1.0,
        },
        {
            "dataset": "d1",
            "strategy": "random",
            "seed": 1,
            "budget": 20,
            "macro_f1": 0.90,
            "runtime_seconds": 1.0,
        },
        {
            "dataset": "d1",
            "strategy": "candidate",
            "seed": 1,
            "budget": 10,
            "macro_f1": 0.80,
            "runtime_seconds": 1.0,
        },
        {
            "dataset": "d1",
            "strategy": "candidate",
            "seed": 1,
            "budget": 20,
            "macro_f1": 0.90,
            "runtime_seconds": 1.0,
        },
    ]
    _write_csv(tmp_path / "metrics.csv", rows)

    report = build_quality_gate_report(tmp_path)

    rates = next(row for row in report["strategy_win_rate_vs_random"] if row["strategy"] == "candidate")
    assert rates["win_rate"] == 0.0
    assert rates["non_loss_rate"] == 1.0
    assert report["quality_gates"]["passed"] is False
    assert report["quality_gates"]["passing_strategies"] == []

    candidate_gate = next(
        check for check in report["quality_gates"]["checks"] if check["name"] == "single_strategy_quality_candidate"
    )
    assert candidate_gate["passed"] is False


def test_quality_gate_fails_when_all_non_random_selections_collapse(tmp_path: Path) -> None:
    rows = []
    selection_rows = []
    for strategy, score in [("random", 0.50), ("entropy", 0.60), ("margin", 0.61), ("badge", 0.62)]:
        rows.append(
            {
                "dataset": "many_class",
                "strategy": strategy,
                "seed": 13,
                "budget": 20,
                "macro_f1": score,
                "runtime_seconds": 1.0,
            }
        )
        selected_ids = ["different"] if strategy == "random" else ["same_a", "same_b"]
        snapshot = {"mode": "single", "strategy": strategy}
        if strategy in {"entropy", "margin", "badge"}:
            snapshot["strategy_diagnostics"] = [
                {
                    "strategy": strategy,
                    "effective_strategy": "coreset_kcenter",
                    "fallback_reason": "cold_start_sparse_probability_support",
                }
            ]
        selection_rows.append(
            {
                "dataset": "many_class",
                "strategy": strategy,
                "seed": 13,
                "budget": 20,
                "selected_ids": json.dumps(selected_ids),
                "scheduler_snapshot": json.dumps(snapshot),
            }
        )
    _write_csv(tmp_path / "metrics.csv", rows)
    _write_csv(tmp_path / "selections.csv", selection_rows)

    report = build_quality_gate_report(tmp_path)

    diff = report["selection_differentiation"]
    assert diff["available"] is True
    assert diff["comparable_group_count"] == 1
    assert diff["collapsed_selected_order_group_count"] == 1
    assert diff["all_comparable_groups_collapsed"] is True
    gate = next(
        check for check in report["quality_gates"]["checks"] if check["name"] == "non_random_strategy_differentiation"
    )
    assert gate["passed"] is False
    assert report["quality_gates"]["passed"] is False


def test_quality_gate_accepts_differentiated_non_random_selections(tmp_path: Path) -> None:
    rows = []
    selection_rows = []
    scores = {"random": 0.50, "entropy": 0.60, "margin": 0.61}
    for strategy, score in scores.items():
        rows.append(
            {
                "dataset": "many_class",
                "strategy": strategy,
                "seed": 13,
                "budget": 20,
                "macro_f1": score,
                "runtime_seconds": 1.0,
            }
        )
        selected_ids = {
            "random": ["r"],
            "entropy": ["a", "b"],
            "margin": ["c", "d"],
        }[strategy]
        selection_rows.append(
            {
                "dataset": "many_class",
                "strategy": strategy,
                "seed": 13,
                "budget": 20,
                "selected_ids": json.dumps(selected_ids),
                "scheduler_snapshot": json.dumps({"mode": "single", "strategy": strategy}),
            }
        )
    _write_csv(tmp_path / "metrics.csv", rows)
    _write_csv(tmp_path / "selections.csv", selection_rows)

    report = build_quality_gate_report(tmp_path)

    diff = report["selection_differentiation"]
    assert diff["comparable_group_count"] == 1
    assert diff["collapsed_selected_order_group_count"] == 0
    gate = next(
        check for check in report["quality_gates"]["checks"] if check["name"] == "non_random_strategy_differentiation"
    )
    assert gate["passed"] is True


def test_quality_gate_fails_when_strategy_pair_collapses_across_all_groups(tmp_path: Path) -> None:
    rows = []
    selection_rows = []
    for seed in [1, 2]:
        for strategy, selected_ids in {
            "random": [f"r{seed}"],
            "entropy": ["same_a", "same_b"],
            "margin": ["same_a", "same_b"],
            "badge": [f"badge_{seed}"],
        }.items():
            rows.append(
                {
                    "dataset": "many_class",
                    "strategy": strategy,
                    "seed": seed,
                    "budget": 20,
                    "macro_f1": 0.60 if strategy != "random" else 0.50,
                    "runtime_seconds": 1.0,
                }
            )
            selection_rows.append(
                {
                    "dataset": "many_class",
                    "strategy": strategy,
                    "seed": seed,
                    "budget": 20,
                    "selected_ids": json.dumps(selected_ids),
                    "scheduler_snapshot": json.dumps({"mode": "single", "strategy": strategy}),
                }
            )
    _write_csv(tmp_path / "metrics.csv", rows)
    _write_csv(tmp_path / "selections.csv", selection_rows)

    report = build_quality_gate_report(tmp_path)

    diff = report["selection_differentiation"]
    assert diff["all_comparable_groups_collapsed"] is False
    assert diff["pairwise_collapsed_pair_count"] == 1
    assert diff["pairwise_collapsed_pairs"][0]["left_strategy"] == "entropy"
    assert diff["pairwise_collapsed_pairs"][0]["right_strategy"] == "margin"
    gate = next(
        check for check in report["quality_gates"]["checks"] if check["name"] == "non_random_strategy_differentiation"
    )
    assert gate["passed"] is False


def test_selection_differentiation_ignores_no_acquisition_budget_rows(tmp_path: Path) -> None:
    rows = []
    selection_rows = []
    for strategy in ["random", "entropy", "margin"]:
        rows.append(
            {
                "dataset": "tiny",
                "strategy": strategy,
                "seed": 1,
                "budget": 9,
                "macro_f1": 0.50,
                "runtime_seconds": 1.0,
            }
        )
        rows.append(
            {
                "dataset": "tiny",
                "strategy": strategy,
                "seed": 1,
                "budget": 12,
                "macro_f1": 0.60 if strategy != "random" else 0.50,
                "runtime_seconds": 1.0,
            }
        )
        selection_rows.append(
            {
                "dataset": "tiny",
                "strategy": strategy,
                "seed": 1,
                "budget": 9,
                "selected_ids": json.dumps([]),
                "scheduler_snapshot": json.dumps({"mode": "single", "no_acquisition_needed": True}),
            }
        )
        selection_rows.append(
            {
                "dataset": "tiny",
                "strategy": strategy,
                "seed": 1,
                "budget": 12,
                "selected_ids": json.dumps({"random": ["r"], "entropy": ["a"], "margin": ["b"]}[strategy]),
                "scheduler_snapshot": json.dumps({"mode": "single", "strategy": strategy}),
            }
        )
    _write_csv(tmp_path / "metrics.csv", rows)
    _write_csv(tmp_path / "selections.csv", selection_rows)

    report = build_quality_gate_report(tmp_path)

    diff = report["selection_differentiation"]
    assert diff["comparable_group_count"] == 1
    assert diff["collapsed_selected_order_group_count"] == 0
    assert diff["pairwise_collapsed_pair_count"] == 0


def test_quality_gate_allows_saturated_final_score_with_positive_aulc_lift(tmp_path: Path) -> None:
    rows = [
        {
            "dataset": "d1",
            "strategy": "random",
            "seed": 1,
            "budget": 10,
            "macro_f1": 0.80,
            "runtime_seconds": 1.0,
        },
        {
            "dataset": "d1",
            "strategy": "random",
            "seed": 1,
            "budget": 20,
            "macro_f1": 0.90,
            "runtime_seconds": 1.0,
        },
        {
            "dataset": "d1",
            "strategy": "candidate",
            "seed": 1,
            "budget": 10,
            "macro_f1": 0.82,
            "runtime_seconds": 1.0,
        },
        {
            "dataset": "d1",
            "strategy": "candidate",
            "seed": 1,
            "budget": 20,
            "macro_f1": 0.90,
            "runtime_seconds": 1.0,
        },
    ]
    _write_csv(tmp_path / "metrics.csv", rows)

    report = build_quality_gate_report(tmp_path)

    final_lift = next(row for row in report["final_budget_lift_vs_random"] if row["strategy"] == "candidate")
    assert final_lift["mean_lift"] == 0.0

    aulc_lift = next(row for row in report["mean_aulc_lift_vs_random"] if row["strategy"] == "candidate")
    assert aulc_lift["mean_aulc_lift"] > 0.0

    rates = next(row for row in report["strategy_win_rate_vs_random"] if row["strategy"] == "candidate")
    assert rates["win_rate"] == 0.5
    assert rates["non_loss_rate"] == 1.0
    assert report["quality_gates"]["passed"] is True
    assert report["quality_gates"]["passing_strategies"] == ["candidate"]


def test_quality_gate_fails_on_partial_random_baseline_coverage(tmp_path: Path) -> None:
    rows = [
        {
            "dataset": "d1",
            "strategy": "random",
            "seed": 1,
            "budget": 10,
            "macro_f1": 0.50,
            "runtime_seconds": 1.0,
        },
        {
            "dataset": "d1",
            "strategy": "entropy",
            "seed": 1,
            "budget": 10,
            "macro_f1": 0.60,
            "runtime_seconds": 1.0,
        },
        {
            "dataset": "d1",
            "strategy": "entropy",
            "seed": 1,
            "budget": 20,
            "macro_f1": 0.70,
            "runtime_seconds": 1.0,
        },
    ]
    _write_csv(tmp_path / "metrics.csv", rows)

    report = build_quality_gate_report(tmp_path)

    assert report["quality_gates"]["passed"] is False
    baseline_gate = next(check for check in report["quality_gates"]["checks"] if check["name"] == "random_baseline_complete")
    assert baseline_gate["passed"] is False


def test_quality_gate_report_keeps_manifest_evidence_and_claim_categories_separate(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "metrics.csv",
        [
            {
                "dataset": "toy",
                "strategy": "random",
                "strategy_family": "sdk",
                "seed": 1,
                "budget": 10,
                "macro_f1": 0.50,
                "runtime_seconds": 0.10,
            },
            {
                "dataset": "toy",
                "strategy": "entropy",
                "strategy_family": "sdk",
                "seed": 1,
                "budget": 10,
                "macro_f1": 0.60,
                "runtime_seconds": 0.20,
            },
            {
                "dataset": "toy",
                "strategy": "manual_entropy",
                "strategy_family": "manual",
                "seed": 1,
                "budget": 10,
                "macro_f1": 0.60,
                "runtime_seconds": 0.15,
            },
            {
                "dataset": "toy",
                "strategy": "modal_formula_entropy",
                "strategy_family": "formula_shim",
                "seed": 1,
                "budget": 10,
                "macro_f1": 0.60,
                "runtime_seconds": 0.16,
            },
        ],
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "preset": "smoke",
                "run_id": "test-run",
                "benchmark_contract": "SDK and formula references use identical probabilities.",
                "datasets": ["toy"],
                "strategies": ["random", "entropy", "manual_entropy", "modal_formula_entropy"],
                "skipped_strategies": {"modal_entropy": "skipped: optional dependency missing"},
            }
        ),
        encoding="utf-8",
    )

    report = build_quality_gate_report(tmp_path)

    assert report["schema_version"] == 2
    assert report["evidence"]["manifest_present"] is True
    assert report["evidence"]["evidence_category"] == "reference_formula_comparison"
    categories = report["evidence"]["strategy_claim_categories"]
    assert categories["sdk_native"] == ["entropy", "random"]
    assert categories["manual_formula_reference"] == ["manual_entropy"]
    assert categories["external_formula_shim"] == ["modal_formula_entropy"]
    assert categories["native_external"] == []
    assert "formula-shim" in report["evidence"]["external_native_status"]["note"].lower()

    claim_gate = next(
        check for check in report["quality_gates"]["checks"] if check["name"] == "strategy_claim_categories_separated"
    )
    assert claim_gate["passed"] is True
    manifest_gate = next(
        check for check in report["quality_gates"]["checks"] if check["name"] == "manifest_evidence_category_present"
    )
    assert manifest_gate["passed"] is True


def test_quality_gate_fails_real_standard_when_calibration_and_coverage_are_missing(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for seed in [13, 21, 34]:
        rows.append(
            {
                "dataset": "banking77",
                "strategy": "random",
                "seed": seed,
                "budget": 100,
                "macro_f1": 0.50,
                "runtime_seconds": 1.0,
            }
        )
        rows.append(
            {
                "dataset": "banking77",
                "strategy": "entropy",
                "seed": seed,
                "budget": 100,
                "macro_f1": 0.60,
                "runtime_seconds": 1.1,
            }
        )
    _write_csv(tmp_path / "metrics.csv", rows)
    _write_csv(
        tmp_path / "full_train_reference.csv",
        [
            {"dataset": "banking77", "seed": seed, "macro_f1": 0.70, "runtime_seconds": 10.0}
            for seed in [13, 21, 34]
        ],
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "preset": "real_medium",
                "run_id": "real-standard",
                "datasets": ["banking77"],
                "strategies": ["random", "entropy"],
                "seeds": [13, 21, 34],
                "seed_count": 3,
                "max_train_samples": 800,
                "max_test_samples": 500,
                "real_evidence_level": "standard",
            }
        ),
        encoding="utf-8",
    )

    report = build_quality_gate_report(tmp_path)

    assert report["quality_gates"]["passed"] is False
    metrics_gate = next(
        check for check in report["quality_gates"]["checks"] if check["name"] == "stage11_real_standard_metrics_present"
    )
    reference_gate = next(
        check
        for check in report["quality_gates"]["checks"]
        if check["name"] == "stage11_real_standard_full_train_calibration_present"
    )
    seed_gate = next(
        check for check in report["quality_gates"]["checks"] if check["name"] == "stage11_real_standard_seed_count"
    )
    assert metrics_gate["passed"] is False
    assert reference_gate["passed"] is False
    assert seed_gate["passed"] is True


def test_quality_gate_does_not_classify_native_external_only_as_formula_comparison(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "metrics.csv",
        [
            {
                "dataset": "toy",
                "strategy": "modal_native_entropy",
                "strategy_family": "native_external",
                "seed": 1,
                "budget": 10,
                "macro_f1": 0.50,
                "runtime_seconds": 0.10,
            },
            {
                "dataset": "toy",
                "strategy": "skactiveml_native_uncertainty",
                "strategy_family": "native_external",
                "seed": 1,
                "budget": 10,
                "macro_f1": 0.60,
                "runtime_seconds": 0.20,
            },
        ],
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "preset": "native_external_smoke",
                "run_id": "native-external",
                "benchmark_claim_category": "native_external_library_workflow_smoke",
                "datasets": ["toy"],
                "strategies": ["modal_native_entropy", "skactiveml_native_uncertainty"],
            }
        ),
        encoding="utf-8",
    )

    report = build_quality_gate_report(tmp_path)

    assert report["evidence"]["evidence_category"] == "native_external_library_workflow_smoke"
    assert report["evidence"]["strategy_claim_categories"]["native_external"] == [
        "modal_native_entropy",
        "skactiveml_native_uncertainty",
    ]


def test_quality_gate_fails_when_runtime_summary_is_not_computable(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "metrics.csv",
        [
            {
                "dataset": "d1",
                "strategy": "random",
                "seed": 1,
                "budget": 10,
                "macro_f1": 0.50,
                "runtime_seconds": "",
            },
            {
                "dataset": "d1",
                "strategy": "entropy",
                "seed": 1,
                "budget": 10,
                "macro_f1": 0.60,
                "runtime_seconds": "",
            },
        ],
    )

    report = build_quality_gate_report(tmp_path)

    runtime_gate = next(check for check in report["quality_gates"]["checks"] if check["name"] == "runtime_summary_present")
    assert runtime_gate["passed"] is False
    assert report["runtime_mean"] == []
    assert report["quality_gates"]["passed"] is False
