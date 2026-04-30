from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


RANDOM_STRATEGY = "random"
METRICS_FILE = "metrics.csv"
SELECTIONS_FILE = "selections.csv"
FULL_TRAIN_REFERENCE_FILE = "full_train_reference.csv"
JSON_REPORT_FILE = "quality_gate.json"
MARKDOWN_REPORT_FILE = "quality_gate.md"
MANIFEST_FILE = "manifest.json"

REQUIRED_COLUMNS = {"dataset", "strategy", "seed", "budget", "macro_f1", "runtime_seconds"}
CALIBRATION_METRICS = ("multiclass_brier_score", "nll", "ece")
REAL_DATASETS = {"banking77", "clinc_oos_imbalanced", "clinc_oos_plus", "dair_ai_emotion"}
REAL_STANDARD_PRESETS = {"real_medium", "real_full"}
REAL_STANDARD_MIN_SEEDS = 3
REAL_STANDARD_REQUIRED_METRICS = (
    "label_coverage_fraction",
    "class_coverage_fraction",
    "zero_recall_class_fraction",
    *CALIBRATION_METRICS,
)
MEANINGFUL_ACQUISITION_MIN_SELECTED = 2
SUMMARY_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_recall",
    "rare_recall",
    *CALIBRATION_METRICS,
    "label_coverage_fraction",
    "class_coverage_fraction",
    "zero_recall_class_fraction",
    "runtime_seconds",
)
COVERAGE_COLUMNS = ("label_coverage_fraction", "class_coverage_fraction")
ZERO_RECALL_COLUMN = "zero_recall_class_fraction"
METADATA_COLUMNS = {"dataset", "strategy", "seed", "budget"}
NUMERIC_COLUMNS = set(SUMMARY_METRICS) | {
    "requested_budget",
    "initial_seed_size",
    "early_macro_f1",
    "accuracy_per_label",
    "macro_f1_per_label",
    "selected_count",
    "rare_selected_count",
    *CALIBRATION_METRICS,
    "aulc_accuracy",
    "aulc_macro_f1",
    "aulc_weighted_f1",
    "aulc_rare_recall",
    "aulc_multiclass_brier_score",
    "aulc_nll",
    "aulc_ece",
    "lift_accuracy_vs_random",
    "lift_macro_f1_vs_random",
    "lift_weighted_f1_vs_random",
    "lift_rare_recall_vs_random",
    "lift_multiclass_brier_score_vs_random",
    "lift_nll_vs_random",
    "lift_ece_vs_random",
}


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        parsed = float(text)
    if not math.isfinite(parsed):
        return None
    return parsed


def _parse_int(value: Any) -> int:
    parsed = _parse_float(value)
    if parsed is None:
        raise ValueError(f"Expected an integer-compatible value, got {value!r}.")
    return int(parsed)


def _normalize_metric_or_metadata(column: str, value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if column in NUMERIC_COLUMNS:
        return _parse_float(text)
    try:
        return _parse_float(text)
    except ValueError:
        return text


def _mean(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(statistics.fmean(finite))


def _std(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    if len(finite) == 1:
        return 0.0
    return float(statistics.stdev(finite))


def _strict_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_json_field(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _strategy_claim_category(strategy: str, family: Any = None) -> str:
    strategy_name = str(strategy)
    family_name = str(family).strip().lower() if family is not None else ""
    if family_name == "formula_shim" or "_formula_" in strategy_name:
        return "external_formula_shim"
    if family_name == "manual" or strategy_name.startswith("manual_"):
        return "manual_formula_reference"
    if family_name in {"external", "native_external"}:
        return "native_external"
    if strategy_name.startswith(("modal_", "skactiveml_")):
        return "native_external"
    if family_name in {"", "sdk", "sdk_native"}:
        return "sdk_native"
    return "unknown"


def _evidence_category(manifest: dict[str, Any] | None, rows: list[dict[str, Any]]) -> str:
    datasets = set(_as_string_list((manifest or {}).get("datasets"))) or {str(row["dataset"]) for row in rows}
    preset = str((manifest or {}).get("preset", "")).lower()
    claim_category = str((manifest or {}).get("benchmark_claim_category", "")).lower()
    has_reference_contract = bool((manifest or {}).get("runnable_strategies") or (manifest or {}).get("skipped_strategies"))
    has_formula_rows = any(
        _strategy_claim_category(str(row["strategy"]), row.get("strategy_family"))
        in {"manual_formula_reference", "external_formula_shim"}
        for row in rows
    )
    if claim_category == "native_external_library_workflow_smoke" or (
        rows
        and all(
            _strategy_claim_category(str(row["strategy"]), row.get("strategy_family")) == "native_external"
            for row in rows
        )
    ):
        return "native_external_library_workflow_smoke"
    if has_reference_contract or has_formula_rows:
        return "reference_formula_comparison"
    if preset.startswith("real") or datasets & REAL_DATASETS:
        return "sdk_native_capped_real_dataset"
    return "sdk_native_synthetic_diagnostic"


def _evidence_summary(input_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_path = input_path / MANIFEST_FILE
    manifest = _read_json(manifest_path) if manifest_path.exists() else None
    strategy_families: dict[str, set[str]] = defaultdict(set)
    categories: dict[str, list[str]] = {
        "sdk_native": [],
        "manual_formula_reference": [],
        "external_formula_shim": [],
        "native_external": [],
        "unknown": [],
    }
    for row in rows:
        strategy = str(row["strategy"])
        family = row.get("strategy_family")
        category = _strategy_claim_category(strategy, family)
        strategy_families[strategy].add(category)
    for strategy, strategy_categories in strategy_families.items():
        if len(strategy_categories) == 1:
            categories[next(iter(strategy_categories))].append(strategy)
        else:
            categories["unknown"].append(strategy)

    categories = {key: sorted(set(values)) for key, values in categories.items()}
    external_skips = (manifest or {}).get("skipped_strategies", {})
    return {
        "manifest_present": manifest is not None,
        "manifest_path": str(manifest_path) if manifest is not None else None,
        "evidence_category": _evidence_category(manifest, rows) if manifest is not None else "metrics_only_no_manifest",
        "manifest": {
            "preset": manifest.get("preset") if manifest else None,
            "run_id": manifest.get("run_id") if manifest else None,
            "artifact_schema_version": manifest.get("artifact_schema_version") if manifest else None,
            "git_dirty": manifest.get("git", {}).get("dirty") if isinstance(manifest, dict) else None,
            "datasets": _as_string_list(manifest.get("datasets")) if manifest else [],
            "strategies": _as_string_list(manifest.get("strategies")) if manifest else [],
            "seed_count": manifest.get("seed_count") if manifest else None,
            "seeds": manifest.get("seeds") if manifest else [],
            "max_train_samples": manifest.get("max_train_samples") if manifest else None,
            "max_test_samples": manifest.get("max_test_samples") if manifest else None,
            "real_evidence_level": manifest.get("real_evidence_level") if manifest else None,
            "allow_uncapped_real_standard": manifest.get("allow_uncapped_real_standard") if manifest else None,
        },
        "strategy_claim_categories": categories,
        "external_native_status": {
            "skipped": external_skips if isinstance(external_skips, dict) else {},
            "note": (
                "Formula-shim or manual-reference rows are not native external-library workflow evidence; "
                "native external rows are reported only in native_external."
            ),
        },
        "claim_boundary": (
            "SDK-native, manual formula-reference, external formula-shim, and native external-library evidence "
            "are categorized separately so formula and native runtime claims are not conflated."
        ),
    }


def _metric_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["strategy"], row["budget"])].append(row)

    summaries: list[dict[str, Any]] = []
    for (dataset, strategy, budget), group_rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "dataset": dataset,
            "strategy": strategy,
            "budget": budget,
            "n": len(group_rows),
        }
        for metric in SUMMARY_METRICS:
            values = [row[metric] for row in group_rows if row.get(metric) is not None]
            if values:
                summary[f"{metric}_mean"] = _mean(values)
                summary[f"{metric}_std"] = _std(values)
        summaries.append(summary)
    return summaries


def _seed_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seeds = sorted({int(row["seed"]) for row in rows})
    rows_by_seed: dict[str, int] = defaultdict(int)
    for row in rows:
        rows_by_seed[str(row["seed"])] += 1
    return {
        "count": len(seeds),
        "seeds": seeds,
        "rows_by_seed": dict(sorted(rows_by_seed.items(), key=lambda item: int(item[0]))),
    }


def _calibration_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_rows = _final_budget_rows(rows)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in final_rows:
        grouped[(row["dataset"], row["strategy"])].append(row)

    summaries: list[dict[str, Any]] = []
    for (dataset, strategy), group_rows in sorted(grouped.items()):
        summary: dict[str, Any] = {"dataset": dataset, "strategy": strategy, "n": len(group_rows)}
        for metric in CALIBRATION_METRICS:
            values = [row[metric] for row in group_rows if row.get(metric) is not None]
            if values:
                summary[f"{metric}_mean"] = _mean(values)
                summary[f"{metric}_std"] = _std(values)
        summaries.append(summary)
    return summaries


def _final_budget_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_budget_by_dataset = {
        dataset: max(row["budget"] for row in dataset_rows)
        for dataset, dataset_rows in _group_by(rows, "dataset").items()
    }
    return [row for row in rows if row["budget"] == max_budget_by_dataset[row["dataset"]]]


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return dict(grouped)


def _random_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], dict[str, Any]]:
    return {
        (row["dataset"], row["seed"], row["budget"]): row
        for row in rows
        if row["strategy"] == RANDOM_STRATEGY
    }


def _final_budget_lift(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_rows = _final_budget_rows(rows)
    random_by_key = _random_lookup(final_rows)
    lifts: list[dict[str, Any]] = []

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    missing = 0
    for row in final_rows:
        if row["strategy"] == RANDOM_STRATEGY:
            continue
        baseline = random_by_key.get((row["dataset"], row["seed"], row["budget"]))
        if baseline is None or baseline.get("macro_f1") is None or row.get("macro_f1") is None:
            missing += 1
            continue
        grouped[(row["dataset"], row["strategy"])].append(row["macro_f1"] - baseline["macro_f1"])

    for (dataset, strategy), values in sorted(grouped.items()):
        lifts.append(
            {
                "dataset": dataset,
                "strategy": strategy,
                "metric": "macro_f1",
                "mean_lift": _mean(values),
                "std_lift": _std(values),
                "n": len(values),
            }
        )
    if missing:
        lifts.append({"dataset": None, "strategy": None, "metric": "macro_f1", "missing_baseline_rows": missing})
    return lifts


def _win_rates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    random_by_key = _random_lookup(rows)
    wins: dict[str, list[float]] = defaultdict(list)
    non_losses: dict[str, list[float]] = defaultdict(list)
    missing: dict[str, int] = defaultdict(int)

    for row in rows:
        strategy = row["strategy"]
        if strategy == RANDOM_STRATEGY:
            continue
        baseline = random_by_key.get((row["dataset"], row["seed"], row["budget"]))
        if baseline is None or baseline.get("macro_f1") is None or row.get("macro_f1") is None:
            missing[strategy] += 1
            continue
        wins[strategy].append(1.0 if row["macro_f1"] > baseline["macro_f1"] else 0.0)
        non_losses[strategy].append(1.0 if row["macro_f1"] >= baseline["macro_f1"] else 0.0)

    strategies = sorted(set(wins) | set(non_losses) | set(missing))
    return [
        {
            "strategy": strategy,
            "metric": "macro_f1",
            "win_rate": _mean(wins.get(strategy, [])),
            "non_loss_rate": _mean(non_losses.get(strategy, [])),
            "comparisons": len(wins.get(strategy, [])),
            "missing_baseline_rows": missing.get(strategy, 0),
        }
        for strategy in strategies
    ]


def _meaningful_acquisition_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    saw_selected_count = False
    for row in rows:
        selected_count = row.get("selected_count")
        if selected_count is None:
            continue
        saw_selected_count = True
        if float(selected_count) >= MEANINGFUL_ACQUISITION_MIN_SELECTED:
            filtered.append(row)
    return filtered if saw_selected_count else list(rows)


def _meaningful_acquisition_win_rates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rates = _win_rates(_meaningful_acquisition_rows(rows))
    return [
        {
            **row,
            "scope": "meaningful_acquisition",
            "min_selected_count": MEANINGFUL_ACQUISITION_MIN_SELECTED,
        }
        for row in rates
    ]


def _final_budget_win_rates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_rows = _final_budget_rows(rows)
    rates = _win_rates(final_rows)
    return [{**row, "scope": "final_budget"} for row in rates]


def _normalized_aulc(rows: list[dict[str, Any]], full_train_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    ref_by_key = {
        (row.get("dataset", ""), _parse_int(row.get("seed", 0))): _parse_float(row.get("macro_f1"))
        for row in full_train_rows
    }
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["strategy"], row["seed"])].append(row)

    aulc_rows: list[dict[str, Any]] = []
    for (dataset, strategy, seed), group_rows in sorted(grouped.items()):
        points = sorted(
            (row["budget"], row["macro_f1"])
            for row in group_rows
            if row.get("macro_f1") is not None
        )
        if not points:
            normalized = None
        elif len(points) == 1:
            normalized = points[0][1]
        else:
            area = 0.0
            for (left_budget, left_value), (right_budget, right_value) in zip(points, points[1:]):
                area += (right_budget - left_budget) * (left_value + right_value) / 2.0
            budget_span = points[-1][0] - points[0][0]
            normalized = area / budget_span if budget_span > 0 else points[-1][1]

        ref_macro_f1 = ref_by_key.get((dataset, seed))
        aulc_rows.append(
            {
                "dataset": dataset,
                "strategy": strategy,
                "seed": seed,
                "metric": "macro_f1",
                "normalized_aulc": normalized,
                "full_train_macro_f1": ref_macro_f1,
                "normalized_aulc_vs_full_train": (
                    normalized / ref_macro_f1 if normalized is not None and ref_macro_f1 and ref_macro_f1 > 0 else None
                ),
            }
        )
    return aulc_rows


def _mean_aulc_lift(aulc_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    random_by_key = {
        (row["dataset"], row["seed"]): row
        for row in aulc_rows
        if row["strategy"] == RANDOM_STRATEGY
    }
    lifts: dict[str, list[float]] = defaultdict(list)
    missing: dict[str, int] = defaultdict(int)

    for row in aulc_rows:
        strategy = row["strategy"]
        if strategy == RANDOM_STRATEGY:
            continue
        baseline = random_by_key.get((row["dataset"], row["seed"]))
        if baseline is None or baseline.get("normalized_aulc") is None or row.get("normalized_aulc") is None:
            missing[strategy] += 1
            continue
        lifts[strategy].append(row["normalized_aulc"] - baseline["normalized_aulc"])

    return [
        {
            "strategy": strategy,
            "metric": "macro_f1",
            "mean_aulc_lift": _mean(lifts.get(strategy, [])),
            "std_aulc_lift": _std(lifts.get(strategy, [])),
            "comparisons": len(lifts.get(strategy, [])),
            "missing_baseline_rows": missing.get(strategy, 0),
        }
        for strategy in sorted(set(lifts) | set(missing))
    ]


def _coverage_zero_recall_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_rows = _final_budget_rows(rows)
    random_by_key = _random_lookup(final_rows)
    grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    missing = 0

    for row in final_rows:
        if row["strategy"] == RANDOM_STRATEGY:
            continue
        baseline = random_by_key.get((row["dataset"], row["seed"], row["budget"]))
        if baseline is None:
            missing += 1
            continue
        delta: dict[str, float] = {}
        for column in COVERAGE_COLUMNS:
            if row.get(column) is not None and baseline.get(column) is not None:
                delta[f"{column}_delta"] = row[column] - baseline[column]
        if row.get(ZERO_RECALL_COLUMN) is not None and baseline.get(ZERO_RECALL_COLUMN) is not None:
            delta[f"{ZERO_RECALL_COLUMN}_delta"] = row[ZERO_RECALL_COLUMN] - baseline[ZERO_RECALL_COLUMN]
        if delta:
            grouped[(row["dataset"], row["strategy"])].append(delta)

    results: list[dict[str, Any]] = []
    for (dataset, strategy), deltas in sorted(grouped.items()):
        result: dict[str, Any] = {"dataset": dataset, "strategy": strategy, "n": len(deltas)}
        for key in sorted({key for delta in deltas for key in delta}):
            result[f"{key}_mean"] = _mean(delta[key] for delta in deltas if key in delta)
        results.append(result)
    if missing:
        results.append({"dataset": None, "strategy": None, "missing_baseline_rows": missing})
    return results


def _runtime_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("runtime_seconds") is not None:
            grouped[(row["dataset"], row["strategy"])].append(row["runtime_seconds"])
    return [
        {
            "dataset": dataset,
            "strategy": strategy,
            "runtime_seconds_mean": _mean(values),
            "runtime_seconds_std": _std(values),
            "n": len(values),
        }
        for (dataset, strategy), values in sorted(grouped.items())
    ]


def _manifest_seed_count(evidence: dict[str, Any], seed_summary: dict[str, Any]) -> int:
    manifest_seed_count = evidence["manifest"].get("seed_count")
    if isinstance(manifest_seed_count, int):
        return manifest_seed_count
    seeds = evidence["manifest"].get("seeds")
    if isinstance(seeds, list):
        return len({str(seed) for seed in seeds})
    return int(seed_summary["count"])


def _is_real_standard_evidence(evidence: dict[str, Any]) -> bool:
    manifest = evidence["manifest"]
    evidence_level = str(manifest.get("real_evidence_level") or "").lower()
    preset = str(manifest.get("preset") or "").lower()
    if evidence_level:
        return evidence_level == "standard"
    return preset in REAL_STANDARD_PRESETS


def _rows_have_required_metrics(rows: list[dict[str, Any]], required_metrics: Iterable[str]) -> bool:
    return all(row.get(metric) is not None for row in rows for metric in required_metrics)


def _full_train_has_required_metrics(
    full_train_rows: list[dict[str, str]],
    required_metrics: Iterable[str],
) -> bool:
    if not full_train_rows:
        return False
    for row in full_train_rows:
        for metric in required_metrics:
            if _parse_float(row.get(metric)) is None:
                return False
    return True


def _effective_strategy_from_snapshot(strategy: str, snapshot_value: Any) -> str | None:
    snapshot = _parse_json_field(snapshot_value)
    if not isinstance(snapshot, dict):
        return strategy
    diagnostics = snapshot.get("strategy_diagnostics")
    if isinstance(diagnostics, list):
        for diagnostic in diagnostics:
            if isinstance(diagnostic, dict) and diagnostic.get("effective_strategy"):
                return str(diagnostic["effective_strategy"])
    return str(snapshot.get("strategy") or strategy)


def _selection_differentiation_summary(input_path: Path) -> dict[str, Any]:
    selections_path = input_path / SELECTIONS_FILE
    if not selections_path.exists():
        return {
            "available": False,
            "path": None,
            "comparable_group_count": 0,
            "collapsed_selected_order_group_count": 0,
            "collapsed_effective_strategy_group_count": 0,
            "pairwise_collapsed_pair_count": 0,
            "pairwise_collapsed_pairs": [],
            "all_comparable_groups_collapsed": False,
            "collapsed_groups": [],
        }

    rows = _read_csv(selections_path)
    grouped: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if str(row.get("strategy", "")) == RANDOM_STRATEGY:
            continue
        try:
            key = (str(row["dataset"]), _parse_int(row["seed"]), _parse_int(row["budget"]))
        except (KeyError, ValueError):
            continue
        grouped[key].append(row)

    comparable_groups = 0
    collapsed_selected = 0
    collapsed_effective = 0
    collapsed_groups: list[dict[str, Any]] = []
    pair_totals: dict[tuple[str, str], int] = defaultdict(int)
    pair_selected_collapses: dict[tuple[str, str], int] = defaultdict(int)
    for (dataset, seed, budget), group_rows in sorted(grouped.items()):
        strategies = sorted({str(row.get("strategy", "")) for row in group_rows if row.get("strategy")})
        if len(strategies) < 2:
            continue
        orders: dict[str, tuple[str, ...]] = {}
        effective_strategies: dict[str, str | None] = {}
        for row in group_rows:
            strategy = str(row.get("strategy", ""))
            selected_ids = _parse_json_field(row.get("selected_ids"))
            if not isinstance(selected_ids, list):
                continue
            snapshot = _parse_json_field(row.get("scheduler_snapshot"))
            if not selected_ids or (isinstance(snapshot, dict) and snapshot.get("no_acquisition_needed") is True):
                continue
            orders[strategy] = tuple(str(sample_id) for sample_id in selected_ids)
            effective_strategies[strategy] = _effective_strategy_from_snapshot(strategy, row.get("scheduler_snapshot"))
        if len(orders) < 2:
            continue
        comparable_groups += 1
        selected_collapsed = len(set(orders.values())) == 1
        for left_index, left_strategy in enumerate(strategies):
            for right_strategy in strategies[left_index + 1 :]:
                if left_strategy not in orders or right_strategy not in orders:
                    continue
                pair_key = (left_strategy, right_strategy)
                pair_totals[pair_key] += 1
                if orders[left_strategy] == orders[right_strategy]:
                    pair_selected_collapses[pair_key] += 1
        effective_values = [value for value in effective_strategies.values() if value is not None]
        effective_collapsed = bool(effective_values) and len(set(effective_values)) == 1
        if selected_collapsed:
            collapsed_selected += 1
        if effective_collapsed:
            collapsed_effective += 1
        if selected_collapsed or effective_collapsed:
            collapsed_groups.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "budget": budget,
                    "strategies": strategies,
                    "selected_order_collapsed": selected_collapsed,
                    "effective_strategy_collapsed": effective_collapsed,
                    "effective_strategies": effective_strategies,
                }
            )

    pairwise_collapsed_pairs = [
        {
            "left_strategy": left,
            "right_strategy": right,
            "collapsed_group_count": pair_selected_collapses[(left, right)],
            "comparable_group_count": total,
        }
        for (left, right), total in sorted(pair_totals.items())
        if total > 0 and pair_selected_collapses[(left, right)] == total
    ]

    return {
        "available": True,
        "path": str(selections_path),
        "comparable_group_count": comparable_groups,
        "collapsed_selected_order_group_count": collapsed_selected,
        "collapsed_effective_strategy_group_count": collapsed_effective,
        "pairwise_collapsed_pair_count": len(pairwise_collapsed_pairs),
        "pairwise_collapsed_pairs": pairwise_collapsed_pairs[:20],
        "all_comparable_groups_collapsed": comparable_groups > 0 and collapsed_selected == comparable_groups,
        "collapsed_groups": collapsed_groups[:20],
    }


def _quality_gates(
    rows: list[dict[str, Any]],
    final_lifts: list[dict[str, Any]],
    win_rates: list[dict[str, Any]],
    meaningful_win_rates: list[dict[str, Any]],
    final_win_rates: list[dict[str, Any]],
    aulc_lifts: list[dict[str, Any]],
    runtime: list[dict[str, Any]],
    evidence: dict[str, Any],
    seed_summary: dict[str, Any],
    full_train_rows: list[dict[str, str]],
    selection_differentiation: dict[str, Any],
) -> dict[str, Any]:
    strategies = sorted({row["strategy"] for row in rows})
    has_random = RANDOM_STRATEGY in strategies
    non_random = [strategy for strategy in strategies if strategy != RANDOM_STRATEGY]
    expected_comparison_count = len([row for row in rows if row["strategy"] != RANDOM_STRATEGY])
    observed_win_comparisons = sum(
        int(row.get("comparisons", 0))
        for row in win_rates
        if row.get("strategy") is not None
    )
    missing_random_rows = sum(
        int(row.get("missing_baseline_rows", 0))
        for row in win_rates
        if row.get("strategy") is not None
    )
    final_lifts_by_strategy: dict[str, list[float]] = defaultdict(list)
    for row in final_lifts:
        if row.get("strategy") is None or row.get("mean_lift") is None:
            continue
        final_lifts_by_strategy[str(row["strategy"])].append(float(row["mean_lift"]))
    aulc_lift_by_strategy = {
        str(row["strategy"]): row.get("mean_aulc_lift")
        for row in aulc_lifts
        if row.get("strategy") is not None
    }
    non_loss_rate_by_strategy = {
        str(row["strategy"]): row.get("non_loss_rate")
        for row in meaningful_win_rates
        if row.get("strategy") is not None
    }
    final_non_loss_rate_by_strategy = {
        str(row["strategy"]): row.get("non_loss_rate")
        for row in final_win_rates
        if row.get("strategy") is not None
    }
    strategies_with_non_negative_lift = sorted(
        strategy
        for strategy in non_random
        if (
            any(lift >= 0.0 for lift in final_lifts_by_strategy.get(strategy, []))
            or (
                aulc_lift_by_strategy.get(strategy) is not None
                and aulc_lift_by_strategy[strategy] >= 0.0
            )
        )
    )
    passing_strategies = [
        strategy
        for strategy in non_random
        if final_lifts_by_strategy.get(strategy)
        and all(lift >= 0.0 for lift in final_lifts_by_strategy[strategy])
        and aulc_lift_by_strategy.get(strategy) is not None
        and aulc_lift_by_strategy[strategy] >= 0.0
        and non_loss_rate_by_strategy.get(strategy) is not None
        and non_loss_rate_by_strategy[strategy] >= (2.0 / 3.0)
        and final_non_loss_rate_by_strategy.get(strategy) is not None
        and final_non_loss_rate_by_strategy[strategy] >= (2.0 / 3.0)
        and (any(lift > 0.0 for lift in final_lifts_by_strategy[strategy]) or aulc_lift_by_strategy[strategy] > 0.0)
    ]
    real_standard = _is_real_standard_evidence(evidence)
    manifest_seed_count = _manifest_seed_count(evidence, seed_summary)
    standard_metrics_present = _rows_have_required_metrics(rows, REAL_STANDARD_REQUIRED_METRICS)
    standard_full_train_metrics_present = _full_train_has_required_metrics(full_train_rows, CALIBRATION_METRICS)

    checks = [
        {
            "name": "random_baseline_present",
            "passed": has_random,
            "detail": "Random baseline rows are required for strategy comparisons.",
        },
        {
            "name": "at_least_one_non_random_strategy",
            "passed": bool(non_random),
            "detail": "Quality gates need at least one strategy to compare against random.",
        },
        {
            "name": "random_baseline_complete",
            "passed": has_random and missing_random_rows == 0 and observed_win_comparisons == expected_comparison_count,
            "detail": "Every non-random dataset/seed/budget row must have a matching random baseline.",
        },
        {
            "name": "single_strategy_quality_candidate",
            "passed": bool(passing_strategies),
            "detail": (
                "At least one strategy must simultaneously have non-negative final macro-F1 lift, "
                "non-negative AULC lift, meaningful-acquisition non-loss-rate >= 2/3, final-budget "
                "non-loss-rate >= 2/3, and at least one positive final macro-F1 or AULC lift versus random. "
                f"Meaningful acquisition rows require selected_count >= {MEANINGFUL_ACQUISITION_MIN_SELECTED} "
                "when selected_count is present."
            ),
        },
        {
            "name": "non_random_non_negative_quality_lift",
            "passed": bool(strategies_with_non_negative_lift),
            "detail": (
                "At least one non-random strategy must have a computable non-negative final macro-F1 "
                "or AULC lift versus matched random."
            ),
        },
        {
            "name": "win_rate_available",
            "passed": any(row.get("win_rate") is not None for row in win_rates),
            "detail": "At least one strategy-vs-random macro-F1 win-rate must be computable.",
        },
        {
            "name": "non_random_strategy_differentiation",
            "passed": (
                not bool(selection_differentiation.get("all_comparable_groups_collapsed"))
                and int(selection_differentiation.get("pairwise_collapsed_pair_count", 0)) == 0
            ),
            "detail": (
                "When selections.csv is available, non-random strategies must not all share the same selected "
                "order in every comparable dataset/seed/budget group, and no strategy pair may be identical "
                "across all comparable groups."
            ),
        },
        {
            "name": "runtime_summary_present",
            "passed": bool(runtime) and all(row.get("runtime_seconds_mean") is not None for row in runtime),
            "detail": "Runtime summary rows with finite mean runtime must be present.",
        },
        {
            "name": "manifest_evidence_category_present",
            "passed": (not evidence["manifest_present"]) or bool(evidence.get("evidence_category")),
            "detail": "When manifest.json exists, the report must expose its evidence category.",
        },
        {
            "name": "strategy_claim_categories_separated",
            "passed": not evidence["strategy_claim_categories"].get("unknown"),
            "detail": (
                "SDK-native, manual formula-reference, external formula-shim, and native external "
                "strategy claims must be categorized without overlap."
            ),
        },
        {
            "name": "stage11_real_standard_seed_count",
            "passed": (not real_standard) or manifest_seed_count >= REAL_STANDARD_MIN_SEEDS,
            "detail": (
                "Stage 11 standard real evidence requires at least three distinct seeds; "
                "real_smoke is smoke-only and not standard evidence."
            ),
        },
        {
            "name": "stage11_real_standard_metrics_present",
            "passed": (not real_standard) or standard_metrics_present,
            "detail": (
                "Stage 11 standard real metrics.csv rows must include finite calibration, "
                "coverage, and zero-recall metrics."
            ),
        },
        {
            "name": "stage11_real_standard_full_train_calibration_present",
            "passed": (not real_standard) or standard_full_train_metrics_present,
            "detail": "Stage 11 standard real full_train_reference.csv rows must include finite calibration metrics.",
        },
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "passing_strategies": passing_strategies,
        "strategies_with_non_negative_lift": strategies_with_non_negative_lift,
    }


def _normalize_metrics_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    if not raw_rows:
        raise ValueError("metrics.csv is empty.")
    missing_columns = sorted(REQUIRED_COLUMNS - set(raw_rows[0]))
    if missing_columns:
        raise ValueError(f"metrics.csv is missing required columns: {', '.join(missing_columns)}")

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row: dict[str, Any] = {
            "dataset": str(raw["dataset"]),
            "strategy": str(raw["strategy"]),
            "seed": _parse_int(raw["seed"]),
            "budget": _parse_int(raw["budget"]),
        }
        for column, value in raw.items():
            if column in row or column in METADATA_COLUMNS:
                continue
            row[column] = _normalize_metric_or_metadata(column, value)
        rows.append(row)
    return rows


def build_quality_gate_report(input_dir: str | Path) -> dict[str, Any]:
    input_path = Path(input_dir)
    metrics_path = input_path / METRICS_FILE
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing required benchmark metrics file: {metrics_path}")

    full_train_path = input_path / FULL_TRAIN_REFERENCE_FILE
    full_train_rows = _read_csv(full_train_path) if full_train_path.exists() else []
    rows = _normalize_metrics_rows(_read_csv(metrics_path))
    evidence = _evidence_summary(input_path, rows)

    summaries = _metric_summary(rows)
    seed_summary = _seed_summary(rows)
    calibration = _calibration_summary(rows)
    final_lifts = _final_budget_lift(rows)
    win_rates = _win_rates(rows)
    meaningful_win_rates = _meaningful_acquisition_win_rates(rows)
    final_win_rates = _final_budget_win_rates(rows)
    aulc_rows = _normalized_aulc(rows, full_train_rows)
    aulc_lifts = _mean_aulc_lift(aulc_rows)
    coverage_zero_recall_deltas = _coverage_zero_recall_deltas(rows)
    runtime = _runtime_summary(rows)
    selection_differentiation = _selection_differentiation_summary(input_path)
    gates = _quality_gates(
        rows,
        final_lifts,
        win_rates,
        meaningful_win_rates,
        final_win_rates,
        aulc_lifts,
        runtime,
        evidence,
        seed_summary,
        full_train_rows,
        selection_differentiation,
    )

    report = {
        "schema_version": 2,
        "input_dir": str(input_path),
        "metrics_csv": str(metrics_path),
        "full_train_reference_csv": str(full_train_path) if full_train_path.exists() else None,
        "row_count": len(rows),
        "datasets": sorted({row["dataset"] for row in rows}),
        "strategies": sorted({row["strategy"] for row in rows}),
        "budgets": sorted({row["budget"] for row in rows}),
        "seed_summary": seed_summary,
        "mean_std_by_dataset_strategy_budget": summaries,
        "calibration_summary_at_final_budget": calibration,
        "final_budget_lift_vs_random": final_lifts,
        "strategy_win_rate_vs_random": win_rates,
        "strategy_meaningful_acquisition_win_rate_vs_random": meaningful_win_rates,
        "strategy_final_budget_win_rate_vs_random": final_win_rates,
        "normalized_aulc": aulc_rows,
        "mean_aulc_lift_vs_random": aulc_lifts,
        "coverage_zero_recall_deltas_at_final_budget": coverage_zero_recall_deltas,
        "runtime_mean": runtime,
        "selection_differentiation": selection_differentiation,
        "evidence": evidence,
        "quality_gates": gates,
    }
    return _strict_json_value(report)


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    gates = report["quality_gates"]
    lines = [
        "# Quality Gate Report",
        "",
        f"- Result: {'PASS' if gates['passed'] else 'FAIL'}",
        f"- Input: `{report['input_dir']}`",
        f"- Rows: {report['row_count']}",
        f"- Datasets: {', '.join(report['datasets'])}",
        f"- Strategies: {', '.join(report['strategies'])}",
        f"- Budgets: {', '.join(str(value) for value in report['budgets'])}",
        f"- Seed count: {report['seed_summary']['count']}",
        f"- Evidence category: `{report['evidence']['evidence_category']}`",
        f"- Real evidence level: `{report['evidence']['manifest'].get('real_evidence_level')}`",
        f"- Manifest: {'present' if report['evidence']['manifest_present'] else 'not present'}",
        "",
        "## Gates",
        "",
        "| Gate | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in gates["checks"]:
        lines.append(f"| {check['name']} | {'PASS' if check['passed'] else 'FAIL'} | {check['detail']} |")

    selection_diff = report.get("selection_differentiation") or {}
    if selection_diff.get("available"):
        lines.extend(
            [
                "",
                "## Selection Differentiation",
                "",
                "| Comparable groups | Collapsed selected-order groups | Collapsed effective-strategy groups | Pairwise collapsed pairs |",
                "| ---: | ---: | ---: | ---: |",
                (
                    f"| {selection_diff.get('comparable_group_count', 0)} | "
                    f"{selection_diff.get('collapsed_selected_order_group_count', 0)} | "
                    f"{selection_diff.get('collapsed_effective_strategy_group_count', 0)} | "
                    f"{selection_diff.get('pairwise_collapsed_pair_count', 0)} |"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Evidence And Claim Boundaries",
            "",
            report["evidence"]["claim_boundary"],
            "",
            "| Category | Strategies |",
            "| --- | --- |",
        ]
    )
    for category, strategies in report["evidence"]["strategy_claim_categories"].items():
        strategy_list = ", ".join(f"`{strategy}`" for strategy in strategies) if strategies else "n/a"
        lines.append(f"| {category} | {strategy_list} |")

    lines.extend(
        [
            "",
            "## Calibration At Final Budget",
            "",
            "| Dataset | Strategy | Brier Mean | NLL Mean | ECE Mean | N |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["calibration_summary_at_final_budget"]:
        lines.append(
            f"| {row['dataset']} | {row['strategy']} | "
            f"{_format_number(row.get('multiclass_brier_score_mean'))} | "
            f"{_format_number(row.get('nll_mean'))} | "
            f"{_format_number(row.get('ece_mean'))} | {row['n']} |"
        )

    lines.extend(
        [
            "",
            "## Final Budget Lift vs Random",
            "",
            "| Dataset | Strategy | Mean Macro-F1 Lift | Std | N |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in report["final_budget_lift_vs_random"]:
        if row.get("strategy") is None:
            continue
        lines.append(
            "| {dataset} | {strategy} | {mean} | {std} | {n} |".format(
                dataset=row["dataset"],
                strategy=row["strategy"],
                mean=_format_number(row.get("mean_lift")),
                std=_format_number(row.get("std_lift")),
                n=row.get("n", 0),
            )
        )

    lines.extend(
        [
            "",
            "## Win Rate vs Random",
            "",
            "| Strategy | Win Rate | Non-Loss Rate | Comparisons | Missing Baselines |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["strategy_win_rate_vs_random"]:
        lines.append(
            f"| {row['strategy']} | {_format_number(row.get('win_rate'))} | "
            f"{_format_number(row.get('non_loss_rate'))} | {row['comparisons']} | "
            f"{row['missing_baseline_rows']} |"
        )

    lines.extend(
        [
            "",
            "## Meaningful Acquisition Win Rate vs Random",
            "",
            "| Strategy | Win Rate | Non-Loss Rate | Comparisons | Min Selected Count | Missing Baselines |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["strategy_meaningful_acquisition_win_rate_vs_random"]:
        lines.append(
            f"| {row['strategy']} | {_format_number(row.get('win_rate'))} | "
            f"{_format_number(row.get('non_loss_rate'))} | {row['comparisons']} | "
            f"{row.get('min_selected_count')} | {row['missing_baseline_rows']} |"
        )

    lines.extend(
        [
            "",
            "## Final Budget Win Rate vs Random",
            "",
            "| Strategy | Win Rate | Non-Loss Rate | Comparisons | Missing Baselines |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["strategy_final_budget_win_rate_vs_random"]:
        lines.append(
            f"| {row['strategy']} | {_format_number(row.get('win_rate'))} | "
            f"{_format_number(row.get('non_loss_rate'))} | {row['comparisons']} | "
            f"{row['missing_baseline_rows']} |"
        )

    lines.extend(
        [
            "",
            "## Mean AULC Lift vs Random",
            "",
            "| Strategy | Mean AULC Lift | Std | Comparisons | Missing Baselines |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["mean_aulc_lift_vs_random"]:
        lines.append(
            f"| {row['strategy']} | {_format_number(row.get('mean_aulc_lift'))} | "
            f"{_format_number(row.get('std_aulc_lift'))} | {row['comparisons']} | "
            f"{row['missing_baseline_rows']} |"
        )

    lines.extend(
        [
            "",
            "## Runtime Mean",
            "",
            "| Dataset | Strategy | Runtime Mean Seconds | Std | N |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in report["runtime_mean"]:
        lines.append(
            f"| {row['dataset']} | {row['strategy']} | "
            f"{_format_number(row.get('runtime_seconds_mean'))} | "
            f"{_format_number(row.get('runtime_seconds_std'))} | {row['n']} |"
        )

    lines.append("")
    return "\n".join(lines)


def write_quality_gate_report(input_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir is not None else input_path
    output_path.mkdir(parents=True, exist_ok=True)

    report = build_quality_gate_report(input_path)
    json_path = output_path / JSON_REPORT_FILE
    markdown_path = output_path / MARKDOWN_REPORT_FILE
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build quality gate JSON and Markdown reports for benchmark metrics.")
    parser.add_argument("input_dir", help="Benchmark output directory containing metrics.csv.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for quality_gate.json and quality_gate.md. Defaults to input_dir.",
    )
    args = parser.parse_args(argv)

    report = write_quality_gate_report(args.input_dir, args.output_dir)
    return 0 if report["quality_gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
