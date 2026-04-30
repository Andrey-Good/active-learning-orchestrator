from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import sdk_first_benchmark as sdk_benchmark


SMOKE_BUDGETS = [12, 24, 36]
FULL_BUDGETS = [16, 32, 48, 64, 96]

FORMULA_EQUIVALENT_STRATEGIES = {
    "entropy": "manual_entropy",
    "margin": "manual_margin",
    "least_confidence": "manual_least_confidence",
    "class_group_balanced_entropy": "manual_class_group_balanced_entropy",
}
CLAIM_CATEGORY_FORMULA_PARITY = "formula_parity"
_PROBABILITY_SUM_REL_TOL = 1e-9
_PROBABILITY_SUM_ABS_TOL = 1e-12


@dataclass(frozen=True)
class ReferenceStrategy:
    name: str
    family: str
    selector: Callable[[Sequence[str], int, Any, int], list[str]]
    skip_reason: str | None = None


def _stable_hash_parts(*parts: object) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _context_model_id(context: Any) -> str:
    model_id = getattr(context, "model_id", None)
    if callable(model_id):
        return str(model_id())
    return "unknown"


def _is_real_number(value: Any) -> bool:
    return not isinstance(value, bool) and (isinstance(value, (int, float)) or isinstance(value, Real))


def validate_probability_rows(probabilities: Any, pool_ids: Sequence[str]) -> list[list[float]]:
    try:
        rows = list(probabilities)
    except TypeError as exc:
        raise ValueError("predict_proba output must be row-like.") from exc

    if len(rows) != len(pool_ids):
        raise ValueError(f"Expected {len(pool_ids)} probability rows, got {len(rows)}.")

    validated: list[list[float]] = []
    expected_width: int | None = None
    for row_index, (sample_id, row) in enumerate(zip(pool_ids, rows)):
        if isinstance(row, (str, bytes)):
            raise ValueError(f"Probability row {row_index} for {sample_id!r} must be a numeric sequence.")
        try:
            values = list(row)
        except TypeError as exc:
            raise ValueError(f"Probability row {row_index} for {sample_id!r} must be a sequence.") from exc
        if not values:
            raise ValueError(f"Probability row {row_index} for {sample_id!r} is empty.")
        if len(values) < 2:
            raise ValueError(f"Probability row {row_index} for {sample_id!r} must have at least 2 columns.")
        if expected_width is None:
            expected_width = len(values)
        elif len(values) != expected_width:
            raise ValueError(
                f"Probability row {row_index} for {sample_id!r} has width {len(values)}; expected {expected_width}."
            )

        cleaned: list[float] = []
        for column_index, value in enumerate(values):
            if not _is_real_number(value):
                raise ValueError(f"Probability value at row {row_index}, column {column_index} must be numeric.")
            probability = float(value)
            if not math.isfinite(probability):
                raise ValueError(f"Probability value at row {row_index}, column {column_index} must be finite.")
            if probability < 0:
                raise ValueError(f"Probability value at row {row_index}, column {column_index} must be non-negative.")
            cleaned.append(probability)

        row_sum = sum(cleaned)
        if row_sum <= 0:
            raise ValueError(f"Probability row {row_index} for {sample_id!r} has non-positive sum.")
        if not math.isclose(row_sum, 1.0, rel_tol=_PROBABILITY_SUM_REL_TOL, abs_tol=_PROBABILITY_SUM_ABS_TOL):
            raise ValueError(f"Probability row {row_index} for {sample_id!r} must sum to 1.0; got {row_sum}.")
        validated.append(cleaned)
    return validated


def normalize_probability_rows(probabilities: Any, pool_ids: Sequence[str]) -> list[list[float]]:
    """Backward-compatible name for strict SDK-style probability validation."""
    return validate_probability_rows(probabilities, pool_ids)


def _validate_formula_probability_rows(probabilities: Sequence[Sequence[float]]) -> list[list[float]]:
    rows = list(probabilities)
    return validate_probability_rows(rows, [f"row-{index}" for index in range(len(rows))])


def entropy_scores(probabilities: Sequence[Sequence[float]]) -> list[float]:
    return [-sum(prob * math.log(prob) for prob in row if prob > 0.0) for row in _validate_formula_probability_rows(probabilities)]


def margin_scores(probabilities: Sequence[Sequence[float]]) -> list[float]:
    scores: list[float] = []
    for row in _validate_formula_probability_rows(probabilities):
        ordered = sorted(row, reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) >= 2 else ordered[0]
        scores.append(-margin)
    return scores


def least_confidence_scores(probabilities: Sequence[Sequence[float]]) -> list[float]:
    return [1.0 - max(row) for row in _validate_formula_probability_rows(probabilities)]


def _manual_tie_key(strategy_name: str, model_id: str, sample_id: str) -> str:
    return _stable_hash_parts("manual-tie", strategy_name, model_id, sample_id)


def _sdk_tie_key(strategy_name: str, model_id: str, sample_id: str) -> str:
    return _stable_hash_parts("tie", strategy_name, model_id, sample_id)


def _top_scored(
    pool_ids: Sequence[str],
    scores: Sequence[float],
    k: int,
    *,
    strategy_name: str,
    model_id: str,
    tie_family: str = "manual",
) -> list[str]:
    if k <= 0:
        return []
    tie_key = _sdk_tie_key if tie_family == "sdk" else _manual_tie_key
    ordered = sorted(
        zip(pool_ids, scores),
        key=lambda pair: (-pair[1], tie_key(strategy_name, model_id, pair[0]), pair[0]),
    )
    return [sample_id for sample_id, _ in ordered[:k]]


def _predict_probabilities(context: Any, pool_ids: Sequence[str]) -> list[list[float]]:
    return validate_probability_rows(context.predict_proba(pool_ids), pool_ids)


def manual_entropy_select(pool_ids: Sequence[str], k: int, context: Any, seed: int) -> list[str]:
    del seed
    probabilities = _predict_probabilities(context, pool_ids)
    return _top_scored(
        pool_ids,
        entropy_scores(probabilities),
        k,
        strategy_name="manual_entropy",
        model_id=_context_model_id(context),
    )


def manual_margin_select(pool_ids: Sequence[str], k: int, context: Any, seed: int) -> list[str]:
    del seed
    probabilities = _predict_probabilities(context, pool_ids)
    return _top_scored(
        pool_ids,
        margin_scores(probabilities),
        k,
        strategy_name="manual_margin",
        model_id=_context_model_id(context),
    )


def manual_least_confidence_select(pool_ids: Sequence[str], k: int, context: Any, seed: int) -> list[str]:
    del seed
    probabilities = _predict_probabilities(context, pool_ids)
    return _top_scored(
        pool_ids,
        least_confidence_scores(probabilities),
        k,
        strategy_name="manual_least_confidence",
        model_id=_context_model_id(context),
    )


def manual_random_select(pool_ids: Sequence[str], k: int, context: Any, seed: int) -> list[str]:
    if k <= 0:
        return []
    model_id = _context_model_id(context)
    pool_key = _stable_hash_parts("manual-pool", seed, *sorted(str(sample_id) for sample_id in pool_ids))
    ordered = sorted(
        (str(sample_id) for sample_id in pool_ids),
        key=lambda sample_id: (_stable_hash_parts("manual_random", seed, model_id, pool_key, sample_id), sample_id),
    )
    return ordered[:k]


def _predicted_class_buckets(
    pool_ids: Sequence[str],
    scores: Sequence[float],
    probabilities: Sequence[Sequence[float]],
    *,
    strategy_name: str,
    model_id: str,
) -> tuple[dict[int, list[tuple[str, float]]], list[int]]:
    buckets: dict[int, list[tuple[str, float]]] = {}
    for sample_id, score, probability in zip(pool_ids, scores, probabilities):
        predicted_class = max(range(len(probability)), key=lambda index: (probability[index], -index))
        buckets.setdefault(predicted_class, []).append((sample_id, score))

    for predicted_class, bucket in buckets.items():
        buckets[predicted_class] = sorted(
            bucket,
            key=lambda pair: (-pair[1], _manual_tie_key(strategy_name, model_id, pair[0]), pair[0]),
        )

    class_order = sorted(buckets, key=lambda predicted_class: (-buckets[predicted_class][0][1], predicted_class))
    return buckets, class_order


def _class_balanced_order(
    buckets: dict[int, list[tuple[str, float]]],
    class_order: Sequence[int],
    k: int,
    pool_ids: Sequence[str],
) -> list[str]:
    selected: list[str] = []
    selected_ids: set[str] = set()
    positions = {predicted_class: 0 for predicted_class in class_order}
    target_k = min(k, len(set(str(sample_id) for sample_id in pool_ids)))

    while len(selected) < target_k:
        added = False
        for predicted_class in class_order:
            bucket = buckets[predicted_class]
            position = positions[predicted_class]
            while position < len(bucket) and bucket[position][0] in selected_ids:
                position += 1
            positions[predicted_class] = position
            if position >= len(bucket):
                continue
            sample_id = bucket[position][0]
            selected.append(sample_id)
            selected_ids.add(sample_id)
            positions[predicted_class] = position + 1
            added = True
            if len(selected) >= target_k:
                break
        if not added:
            break
    return selected


def manual_class_group_balanced_entropy_select(pool_ids: Sequence[str], k: int, context: Any, seed: int) -> list[str]:
    del seed
    if k <= 0 or not pool_ids:
        return []
    probabilities = _predict_probabilities(context, pool_ids)
    scores = entropy_scores(probabilities)
    model_id = _context_model_id(context)
    buckets, class_order = _predicted_class_buckets(
        pool_ids,
        scores,
        probabilities,
        strategy_name="manual_class_group_balanced_entropy",
        model_id=model_id,
    )
    target_k = min(k, len(set(str(sample_id) for sample_id in pool_ids)))
    class_balanced = _class_balanced_order(buckets, class_order, target_k, pool_ids)
    samples = context.get_samples(pool_ids)
    groups_by_id = {
        sample_id: ("sample", sample_id) if getattr(sample, "group_id", None) is None else ("group", str(sample.group_id))
        for sample_id, sample in zip(pool_ids, samples)
    }

    selected: list[str] = []
    selected_ids: set[str] = set()
    selected_groups: set[tuple[str, str]] = set()

    while len(selected) < target_k:
        added = False
        for predicted_class in class_order:
            candidate = None
            for sample_id, _ in buckets[predicted_class]:
                if sample_id in selected_ids:
                    continue
                group_key = groups_by_id[sample_id]
                if group_key in selected_groups:
                    continue
                candidate = sample_id
                break
            if candidate is None:
                continue
            selected.append(candidate)
            selected_ids.add(candidate)
            selected_groups.add(groups_by_id[candidate])
            added = True
            if len(selected) >= target_k:
                return selected
        if not added:
            break

    for sample_id in class_balanced:
        if len(selected) >= target_k:
            break
        if sample_id not in selected_ids:
            selected.append(sample_id)
            selected_ids.add(sample_id)
    return selected


def _modal_formula_uncertainty_select(method: str) -> Callable[[Sequence[str], int, Any, int], list[str]]:
    def select(pool_ids: Sequence[str], k: int, context: Any, seed: int) -> list[str]:
        del seed
        probabilities = _predict_probabilities(context, pool_ids)
        if method == "entropy":
            scores = entropy_scores(probabilities)
        elif method == "margin":
            scores = margin_scores(probabilities)
        elif method == "uncertainty":
            scores = least_confidence_scores(probabilities)
        else:
            raise ValueError(f"Unsupported documented modAL formula method: {method}")
        return _top_scored(
            pool_ids,
            scores,
            k,
            strategy_name=f"modal_formula_{method}",
            model_id=_context_model_id(context),
        )

    return select


def _skactiveml_formula_uncertainty_select(method: str) -> Callable[[Sequence[str], int, Any, int], list[str]]:
    def select(pool_ids: Sequence[str], k: int, context: Any, seed: int) -> list[str]:
        del seed
        probabilities = _predict_probabilities(context, pool_ids)
        if method == "entropy":
            scores = entropy_scores(probabilities)
        elif method == "margin":
            scores = [1.0 + score for score in margin_scores(probabilities)]
        elif method == "least_confidence":
            scores = least_confidence_scores(probabilities)
        else:
            raise ValueError(f"Unsupported documented scikit-activeml formula method: {method}")
        return _top_scored(
            pool_ids,
            scores,
            k,
            strategy_name=f"skactiveml_formula_{method}",
            model_id=_context_model_id(context),
        )

    return select


def external_adapter_status() -> dict[str, str]:
    return {
        "modAL": "available" if importlib.util.find_spec("modAL") is not None else "skipped: modAL is not importable",
        "skactiveml": (
            "available" if importlib.util.find_spec("skactiveml") is not None else "skipped: skactiveml is not importable"
        ),
    }


def reference_strategies(include_external: bool = True) -> dict[str, ReferenceStrategy]:
    strategies = {
        "manual_entropy": ReferenceStrategy("manual_entropy", "manual", manual_entropy_select),
        "manual_margin": ReferenceStrategy("manual_margin", "manual", manual_margin_select),
        "manual_least_confidence": ReferenceStrategy(
            "manual_least_confidence",
            "manual",
            manual_least_confidence_select,
        ),
        "manual_class_group_balanced_entropy": ReferenceStrategy(
            "manual_class_group_balanced_entropy",
            "manual",
            manual_class_group_balanced_entropy_select,
        ),
        "manual_random": ReferenceStrategy("manual_random", "manual", manual_random_select),
    }
    if not include_external:
        return strategies

    strategies.update(
        {
            "modal_formula_entropy": ReferenceStrategy(
                "modal_formula_entropy",
                "formula_shim",
                _modal_formula_uncertainty_select("entropy"),
            ),
            "modal_formula_margin": ReferenceStrategy(
                "modal_formula_margin",
                "formula_shim",
                _modal_formula_uncertainty_select("margin"),
            ),
            "modal_formula_uncertainty": ReferenceStrategy(
                "modal_formula_uncertainty",
                "formula_shim",
                _modal_formula_uncertainty_select("uncertainty"),
            ),
            "skactiveml_formula_entropy": ReferenceStrategy(
                "skactiveml_formula_entropy",
                "formula_shim",
                _skactiveml_formula_uncertainty_select("entropy"),
            ),
            "skactiveml_formula_margin": ReferenceStrategy(
                "skactiveml_formula_margin",
                "formula_shim",
                _skactiveml_formula_uncertainty_select("margin"),
            ),
            "skactiveml_formula_least_confidence": ReferenceStrategy(
                "skactiveml_formula_least_confidence",
                "formula_shim",
                _skactiveml_formula_uncertainty_select("least_confidence"),
            ),
        }
    )
    return strategies


def sdk_reference_strategies() -> dict[str, ReferenceStrategy]:
    specs = sdk_benchmark.strategy_specs()

    def make_selector(strategy_name: str) -> Callable[[Sequence[str], int, Any, int], list[str]]:
        def select(pool_ids: Sequence[str], k: int, context: Any, seed: int) -> list[str]:
            del seed
            scheduler = sdk_benchmark.StrategyScheduler(specs[strategy_name].scheduler_config)
            selected, _ = scheduler.select_batch(pool_ids, k, context, state={})
            return selected

        return select

    names = [
        "random",
        "entropy",
        "margin",
        "least_confidence",
        "class_group_balanced_entropy",
        "mix_interleaved_class_group_random",
    ]
    return {name: ReferenceStrategy(name, "sdk", make_selector(name)) for name in names}


def compute_selection_diagnostics(
    selected_ids: Sequence[str],
    labeled_ids: Sequence[str],
    dataset: sdk_benchmark.BenchmarkDataset,
) -> dict[str, Any]:
    return sdk_benchmark.compute_selection_diagnostics(selected_ids, labeled_ids, dataset)


def train_and_evaluate(
    model: Any,
    dataset: sdk_benchmark.BenchmarkDataset,
    labeled_ids: Sequence[str],
    test_ids: Sequence[str],
) -> dict[str, float]:
    return sdk_benchmark.train_and_evaluate(model, dataset, labeled_ids, test_ids)


def run_one_curve(
    dataset: sdk_benchmark.BenchmarkDataset,
    strategy: ReferenceStrategy,
    budgets: Sequence[int],
    seed: int,
    initial_seed_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, list[str]]]:
    sample_by_id = {sample.sample_id: sample for sample in dataset.samples}
    train_ids = sorted(sample.sample_id for sample in dataset.samples if sample.split == "train")
    test_ids = sorted(sample.sample_id for sample in dataset.samples if sample.split == "test")
    max_budget = min(max(budgets), len(train_ids))
    usable_budgets = [budget for budget in sorted(set(budgets)) if initial_seed_size <= budget <= max_budget]

    labeled_ids = sdk_benchmark.choose_initial_seed(dataset, train_ids, initial_seed_size, seed)
    provider = sdk_benchmark.InMemoryBenchmarkProvider([sample for sample in dataset.samples if sample.split == "train"])
    model = sdk_benchmark.SklearnTextBenchmarkAdapter(dataset.labels, seed)
    label_schema = sdk_benchmark.LabelSchema(task="text_classification", labels=dataset.labels)

    metrics_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    selections_by_budget: dict[int, list[str]] = {}

    for budget in usable_budgets:
        before_select = time.perf_counter()
        train_and_evaluate(model, dataset, labeled_ids, test_ids)
        pool_ids = [sample_id for sample_id in train_ids if sample_id not in set(labeled_ids)]
        to_select = min(budget - len(labeled_ids), len(pool_ids))
        selected_ids: list[str] = []
        if to_select > 0:
            context = sdk_benchmark.SelectionContext(
                provider=provider,
                model=model,
                label_schema=label_schema,
                prediction_cache=None,
                embedding_cache=None,
                labeled_ids=labeled_ids,
                last_metrics={},
            )
            selected_ids = strategy.selector(pool_ids, to_select, context, seed)
            labeled_ids.extend(selected_ids)

        metrics = train_and_evaluate(model, dataset, labeled_ids, test_ids)
        elapsed_seconds = time.perf_counter() - before_select
        diagnostics = compute_selection_diagnostics(selected_ids, labeled_ids, dataset)
        coverage_metrics = sdk_benchmark.compute_label_coverage_metrics(dataset, labeled_ids, selected_ids)
        rare_selected = (
            sum(1 for sample_id in selected_ids if sample_by_id[sample_id].label == dataset.rare_label)
            if dataset.rare_label is not None
            else 0
        )
        selections_by_budget[len(labeled_ids)] = list(selected_ids)

        metrics_rows.append(
            {
                "dataset": dataset.name,
                "strategy": strategy.name,
                "strategy_family": strategy.family,
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
                "early_macro_f1": metrics["macro_f1"],
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
                "strategy_family": strategy.family,
                "seed": seed,
                "budget": len(labeled_ids),
                "selected_ids": json.dumps(selected_ids, sort_keys=True),
                "selected_label_counts": json.dumps(diagnostics["selected_label_counts"], sort_keys=True),
                "cumulative_label_counts": json.dumps(diagnostics["cumulative_label_counts"], sort_keys=True),
                "selected_group_counts": json.dumps(diagnostics["selected_group_counts"], sort_keys=True),
                "duplicate_selected_count": diagnostics["duplicate_selected_count"],
                "top_group_fraction": diagnostics["top_group_fraction"],
                "group_hhi": diagnostics["group_hhi"],
            }
        )

    return metrics_rows, selection_rows, selections_by_budget


def add_curve_metrics(metrics_rows: list[dict[str, Any]]) -> None:
    sdk_benchmark.add_curve_metrics(metrics_rows)
    by_curve: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics_rows:
        by_curve[(row["dataset"], row["strategy"], int(row["seed"]))].append(row)
    for rows in by_curve.values():
        rows.sort(key=lambda row: int(row["budget"]))
        early_value = float(rows[0]["macro_f1"]) if rows else float("nan")
        for row in rows:
            row["early_macro_f1"] = early_value
            row["aulc_macro_f1"] = row.get("aulc_macro_f1", float("nan"))


def build_equivalence_rows(curve_selections: dict[tuple[str, str, int], dict[int, list[str]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sdk_name, manual_name in FORMULA_EQUIVALENT_STRATEGIES.items():
        for dataset, strategy, seed in sorted(curve_selections):
            if strategy != sdk_name:
                continue
            manual_key = (dataset, manual_name, seed)
            if manual_key not in curve_selections:
                continue
            for budget, sdk_selected in sorted(curve_selections[(dataset, strategy, seed)].items()):
                manual_selected = curve_selections[manual_key].get(budget, [])
                sdk_set = set(sdk_selected)
                manual_set = set(manual_selected)
                union = sdk_set | manual_set
                intersection = sdk_set & manual_set
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "budget": budget,
                        "sdk_strategy": sdk_name,
                        "reference_strategy": manual_name,
                        "sdk_selected_ids": json.dumps(sdk_selected, sort_keys=True),
                        "reference_selected_ids": json.dumps(manual_selected, sort_keys=True),
                        "overlap_count": len(intersection),
                        "sdk_count": len(sdk_set),
                        "reference_count": len(manual_set),
                        "jaccard": len(intersection) / len(union) if union else 1.0,
                        "exact_order_match": sdk_selected == manual_selected,
                    }
                )
    return rows


def write_summary(
    output_dir: Path,
    metrics_rows: Sequence[dict[str, Any]],
    equivalence_rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    best_by_dataset: dict[str, dict[str, Any]] = {}
    for row in metrics_rows:
        dataset = str(row["dataset"])
        current = best_by_dataset.get(dataset)
        if current is None or float(row["macro_f1"]) > float(current["macro_f1"]):
            best_by_dataset[dataset] = row

    equivalence_summary = {
        "mean_jaccard": (
            sum(float(row["jaccard"]) for row in equivalence_rows) / len(equivalence_rows)
            if equivalence_rows
            else None
        ),
        "exact_order_matches": sum(1 for row in equivalence_rows if row["exact_order_match"]),
        "rows": len(equivalence_rows),
    }
    sdk_benchmark.write_strict_json(
        output_dir / "summary.json",
        {
            "manifest": manifest,
            "best_macro_f1_by_dataset": best_by_dataset,
            "formula_equivalence": equivalence_summary,
            "row_counts": {"metrics": len(metrics_rows), "equivalence": len(equivalence_rows)},
        },
    )

    lines = [
        "# Reference Strategy Benchmark Summary",
        "",
        f"- Run id: `{manifest['run_id']}`",
        f"- Datasets: {', '.join(manifest['datasets'])}",
        f"- Strategies: {', '.join(manifest['strategies'])}",
        f"- Budgets: {', '.join(str(value) for value in manifest['budgets'])}",
        f"- Seeds: {', '.join(str(value) for value in manifest['seeds'])}",
        f"- Mean SDK/manual formula-equivalence Jaccard: {equivalence_summary['mean_jaccard']}",
        "",
        "## Best Macro-F1 By Dataset",
        "",
        "| Dataset | Strategy | Family | Seed | Budget | Macro-F1 | Accuracy |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset, row in sorted(best_by_dataset.items()):
        lines.append(
            f"| {dataset} | {row['strategy']} | {row['strategy_family']} | {row['seed']} | {row['budget']} | "
            f"{float(row['macro_f1']):.4f} | {float(row['accuracy']):.4f} |"
        )
    lines.extend(
        [
            "",
            "Artifacts in this directory:",
            "",
            "- `metrics.csv`: learning-curve quality metrics, AULC, early macro-F1, rare recall, and runtime.",
            "- `selections.csv`: selected ids and group concentration diagnostics.",
            "- `equivalence.csv`: formula-equivalent SDK/manual overlap, Jaccard, and exact order diagnostics.",
            "- `external_adapters.json`: optional external-library availability and skip reasons.",
            "- `manifest.json`: run configuration and benchmark contract.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_csv_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare SDK strategies to manual and optional external references.")
    parser.add_argument("--preset", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names. Defaults depend on preset.")
    parser.add_argument("--strategies", default=None, help="Comma-separated strategy names.")
    parser.add_argument("--budgets", default=None, help="Comma-separated cumulative label budgets.")
    parser.add_argument("--seeds", default="13", help="Comma-separated integer seeds.")
    parser.add_argument("--initial-seed-size", type=int, default=9)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty benchmark output directory.")
    parser.add_argument(
        "--include-external",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include documented external-library formula shims. "
            "This compatibility flag does not make the harness call external query/scorer APIs."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    sdk_benchmark.ensure_benchmark_dependencies()

    sdk_strategies = sdk_reference_strategies()
    reference = reference_strategies(include_external=args.include_external)
    all_strategies = {**sdk_strategies, **reference}

    default_smoke_strategies = [
        "random",
        "manual_random",
        "entropy",
        "manual_entropy",
        "modal_formula_entropy",
        "skactiveml_formula_entropy",
        "margin",
        "manual_margin",
        "modal_formula_margin",
        "skactiveml_formula_margin",
        "least_confidence",
        "manual_least_confidence",
        "modal_formula_uncertainty",
        "skactiveml_formula_least_confidence",
        "class_group_balanced_entropy",
        "manual_class_group_balanced_entropy",
        "mix_interleaved_class_group_random",
    ]
    datasets = (
        parse_csv_list(args.datasets)
        if args.datasets
        else (["separable_topics", "rare_class_trap"] if args.preset == "smoke" else list(sdk_benchmark.DATASET_BUILDERS))
    )
    default_strategies = default_smoke_strategies if args.preset == "smoke" else list(all_strategies)
    if not args.include_external:
        default_strategies = [name for name in default_strategies if name in all_strategies]
    strategies = parse_csv_list(args.strategies) if args.strategies else default_strategies
    budgets = parse_int_list(args.budgets) if args.budgets else (SMOKE_BUDGETS if args.preset == "smoke" else FULL_BUDGETS)
    seeds = parse_int_list(args.seeds)

    unknown_datasets = sorted(set(datasets) - set(sdk_benchmark.DATASET_BUILDERS))
    unknown_strategies = sorted(set(strategies) - set(all_strategies))
    if unknown_datasets:
        raise SystemExit(f"Unknown datasets: {', '.join(unknown_datasets)}")
    if unknown_strategies:
        raise SystemExit(f"Unknown strategies: {', '.join(unknown_strategies)}")
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer seed.")
    max_label_count = max(len(sdk_benchmark.DATASET_BUILDERS[name](seeds[0]).labels) for name in datasets)
    if args.initial_seed_size < max_label_count:
        raise SystemExit("--initial-seed-size must include at least one sample per class.")

    runnable_strategies = [name for name in strategies if all_strategies[name].skip_reason is None]
    skipped_strategies = {
        name: all_strategies[name].skip_reason
        for name in strategies
        if all_strategies[name].skip_reason is not None
    }
    if not runnable_strategies:
        raise SystemExit("No runnable strategies after optional external-library skips.")

    run_id = sdk_benchmark.make_run_id()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else sdk_benchmark.default_output_dir_for_preset(f"reference_{args.preset}", run_id)
    )
    sdk_benchmark.prepare_output_dir(output_dir, overwrite=args.overwrite)
    started = time.perf_counter()
    metrics_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    curve_selections: dict[tuple[str, str, int], dict[int, list[str]]] = {}

    for seed in seeds:
        for dataset_name in datasets:
            dataset = sdk_benchmark.DATASET_BUILDERS[dataset_name](seed)
            validation_rows.append(sdk_benchmark.validate_acquisition_surface(dataset))
            for strategy_name in runnable_strategies:
                curve_metrics, curve_selection_rows, selections_by_budget = run_one_curve(
                    dataset,
                    all_strategies[strategy_name],
                    budgets,
                    seed,
                    args.initial_seed_size,
                )
                metrics_rows.extend(curve_metrics)
                selection_rows.extend(curve_selection_rows)
                curve_selections[(dataset.name, strategy_name, seed)] = selections_by_budget

    add_curve_metrics(metrics_rows)
    equivalence_rows = build_equivalence_rows(curve_selections)
    metrics_rows.sort(key=lambda row: (row["dataset"], row["seed"], row["strategy"], int(row["budget"])))
    selection_rows.sort(key=lambda row: (row["dataset"], row["seed"], row["strategy"], int(row["budget"])))
    equivalence_rows.sort(key=lambda row: (row["dataset"], row["seed"], row["sdk_strategy"], int(row["budget"])))

    artifact_names = {
        "metrics_csv": "metrics.csv",
        "selections_csv": "selections.csv",
        "equivalence_csv": "equivalence.csv",
        "external_adapters_json": "external_adapters.json",
        "manifest_json": "manifest.json",
        "summary_json": "summary.json",
        "summary_md": "summary.md",
        "validation_json": "validation.json",
    }
    manifest = {
        **sdk_benchmark.collect_reproducibility_metadata(raw_argv),
        "run_id": run_id,
        "preset": args.preset,
        "benchmark_claim_category": CLAIM_CATEGORY_FORMULA_PARITY,
        "datasets": datasets,
        "strategies": strategies,
        "runnable_strategies": runnable_strategies,
        "skipped_strategies": skipped_strategies,
        "budgets": budgets,
        "seeds": seeds,
        "initial_seed_size": args.initial_seed_size,
        "include_external": args.include_external,
        "include_external_formula_shims": args.include_external,
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": artifact_names,
        "artifact_names": sorted(set(artifact_names.values())),
        "benchmark_contract": (
            "SDK and reference strategies use the same trained sklearn adapter, pool order, seed, budgets, "
            "and SelectionContext predict_proba outputs. Manual probability rows are strictly validated like SDK "
            "uncertainty strategies: row-like output, one row per pool id, consistent width, at least two finite "
            "non-negative numeric probabilities, and row sums to 1.0 within SDK tolerance; rows are not renormalized. "
            "Manual tie-breaking is deterministic and intentionally benchmark-local. "
            "manual_random remains a stochastic/hash baseline with a different deterministic contract than SDK random "
            "and is excluded from formula-equivalence diagnostics. "
            "modal_formula_* and skactiveml_formula_* rows are documented-formula shims only; this harness records "
            "real external-library importability in external_adapters.json but does not call external scorer or query APIs."
        ),
    }

    sdk_benchmark.write_csv(output_dir / "metrics.csv", metrics_rows)
    sdk_benchmark.write_csv(output_dir / "selections.csv", selection_rows)
    sdk_benchmark.write_csv(output_dir / "equivalence.csv", equivalence_rows)
    sdk_benchmark.write_strict_json(output_dir / "manifest.json", manifest)
    sdk_benchmark.write_strict_json(output_dir / "external_adapters.json", external_adapter_status())
    sdk_benchmark.write_strict_json(
        output_dir / "validation.json",
        {"run_id": run_id, "datasets_checked": len(validation_rows), "checks": validation_rows},
    )
    write_summary(output_dir, metrics_rows, equivalence_rows, manifest)

    print(f"Wrote reference benchmark artifacts to {output_dir.resolve()}")
    print(f"Metrics rows: {len(metrics_rows)}")
    print(f"Selection rows: {len(selection_rows)}")
    print(f"Equivalence rows: {len(equivalence_rows)}")
    if skipped_strategies:
        print(f"Skipped optional strategies: {json.dumps(skipped_strategies, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileExistsError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
