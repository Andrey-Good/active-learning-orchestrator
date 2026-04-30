from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from active_learning_sdk.configs import LabelSchema, SchedulerConfig
from active_learning_sdk.cache import InMemoryCacheStore, PredictionCache
from active_learning_sdk.engine import SelectionContext, StrategyScheduler
from active_learning_sdk.types import DataSample


LABELS = ["alpha", "beta", "gamma"]
SUPPORTED_STRATEGIES = ("entropy", "margin", "least_confidence")
_PROBABILITY_SUM_REL_TOL = 1e-9
_PROBABILITY_SUM_ABS_TOL = 1e-12

CAVEATS = [
    "This is a microbenchmark of acquisition selection only; it does not measure labeling, persistence, training, or end-to-end project orchestration.",
    "The workload uses frozen probabilities rather than a learned model, so quality means formula-level selected-order parity, not downstream macro-F1.",
    "Manual probability rows are strictly validated with the SDK uncertainty contract and are not normalized from logits or count-like rows.",
    "The SDK path warms an in-memory PredictionCache and scheduler capability cache before timing, matching the manual path's precomputed probability map and isolating steady-state scheduler/formula overhead from model/provider round-trips.",
    "Timing ratios are local-process diagnostics and should be read as overhead evidence, not portable latency guarantees.",
    "The candidate pool is intentionally tiny and deterministic; it can expose abstraction overhead but cannot prove broad real-world performance.",
    "Direct modAL and scikit-activeml query-level comparisons are not implemented in this harness; adapter status is an importability check only.",
]


@dataclass(frozen=True)
class AuditCandidate:
    sample_id: str
    text: str
    label: str
    group_id: str
    probabilities: tuple[float, ...]


AUDIT_CANDIDATES = [
    AuditCandidate("audit-001", "audit-001", "alpha", "g01", (0.340, 0.331, 0.329)),
    AuditCandidate("audit-002", "audit-002", "beta", "g02", (0.360, 0.321, 0.319)),
    AuditCandidate("audit-003", "audit-003", "gamma", "g03", (0.390, 0.306, 0.304)),
    AuditCandidate("audit-004", "audit-004", "alpha", "g04", (0.420, 0.291, 0.289)),
    AuditCandidate("audit-005", "audit-005", "beta", "g05", (0.500, 0.489, 0.011)),
    AuditCandidate("audit-006", "audit-006", "gamma", "g06", (0.510, 0.478, 0.012)),
    AuditCandidate("audit-007", "audit-007", "alpha", "g07", (0.520, 0.469, 0.011)),
    AuditCandidate("audit-008", "audit-008", "beta", "g08", (0.400, 0.348, 0.252)),
    AuditCandidate("audit-009", "audit-009", "gamma", "g09", (0.410, 0.338, 0.252)),
    AuditCandidate("audit-010", "audit-010", "alpha", "g10", (0.550, 0.226, 0.224)),
    AuditCandidate("audit-011", "audit-011", "beta", "g11", (0.600, 0.201, 0.199)),
    AuditCandidate("audit-012", "audit-012", "gamma", "g12", (0.700, 0.199, 0.101)),
]


class AuditProvider:
    def __init__(self, candidates: Sequence[AuditCandidate]) -> None:
        self._candidates = {candidate.sample_id: candidate for candidate in candidates}

    def get_sample(self, sample_id: str) -> DataSample:
        candidate = self._candidates[str(sample_id)]
        return DataSample(
            sample_id=candidate.sample_id,
            data={"text": candidate.text},
            meta={"label": candidate.label, "split": "audit"},
            group_id=candidate.group_id,
        )

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        return [self.get_sample(sample_id) for sample_id in sample_ids]

    def get_texts(self, sample_ids: Sequence[str]) -> list[str]:
        return [self._candidates[str(sample_id)].text for sample_id in sample_ids]


class FrozenProbabilityModel:
    def __init__(self, candidates: Sequence[AuditCandidate]) -> None:
        self._probabilities_by_text = {candidate.text: candidate.probabilities for candidate in candidates}

    def get_model_id(self) -> str:
        return "audit-frozen-probabilities-v1"

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        return [list(self._probabilities_by_text[str(text)]) for text in texts]


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
            if probability < 0.0:
                raise ValueError(f"Probability value at row {row_index}, column {column_index} must be non-negative.")
            cleaned.append(probability)

        row_sum = sum(cleaned)
        if row_sum <= 0.0:
            raise ValueError(f"Probability row {row_index} for {sample_id!r} has non-positive sum.")
        if not math.isclose(row_sum, 1.0, rel_tol=_PROBABILITY_SUM_REL_TOL, abs_tol=_PROBABILITY_SUM_ABS_TOL):
            raise ValueError(f"Probability row {row_index} for {sample_id!r} must sum to 1.0; got {row_sum}.")
        validated.append(cleaned)
    return validated


def normalize_probability_rows(probabilities: Any, pool_ids: Sequence[str]) -> list[list[float]]:
    """Backward-compatible name for strict SDK-style probability validation."""
    return validate_probability_rows(probabilities, pool_ids)


def entropy_score(row: Sequence[float]) -> float:
    return -sum(probability * math.log(probability) for probability in row if probability > 0.0)


def margin_score(row: Sequence[float]) -> float:
    ordered = sorted(row, reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) >= 2 else ordered[0]
    return -margin


def least_confidence_score(row: Sequence[float]) -> float:
    return 1.0 - max(row)


def score_function(strategy: str) -> Callable[[Sequence[float]], float]:
    if strategy == "entropy":
        return entropy_score
    if strategy == "margin":
        return margin_score
    if strategy == "least_confidence":
        return least_confidence_score
    raise ValueError(f"Unsupported strategy: {strategy}")


def probability_map(candidates: Sequence[AuditCandidate] = AUDIT_CANDIDATES) -> dict[str, tuple[float, ...]]:
    return {candidate.sample_id: candidate.probabilities for candidate in candidates}


def manual_select(
    pool_ids: Sequence[str],
    k: int,
    probabilities_by_id: dict[str, Sequence[float]],
    strategy: str,
) -> list[str]:
    if k <= 0:
        return []
    scorer = score_function(strategy)
    rows = normalize_probability_rows([probabilities_by_id[sample_id] for sample_id in pool_ids], pool_ids)
    scored = [(sample_id, scorer(row)) for sample_id, row in zip(pool_ids, rows)]
    ordered = sorted(scored, key=lambda pair: (-pair[1], pair[0]))
    return [sample_id for sample_id, _ in ordered[:k]]


def build_context(
    candidates: Sequence[AuditCandidate] = AUDIT_CANDIDATES,
    *,
    use_prediction_cache: bool = True,
) -> SelectionContext:
    return SelectionContext(
        provider=AuditProvider(candidates),
        model=FrozenProbabilityModel(candidates),
        label_schema=LabelSchema(task="text_classification", labels=LABELS),
        prediction_cache=PredictionCache(InMemoryCacheStore()) if use_prediction_cache else None,
        embedding_cache=None,
        labeled_ids=[],
        last_metrics={},
    )


def sdk_select(
    pool_ids: Sequence[str],
    k: int,
    strategy: str,
    context: SelectionContext | None = None,
    scheduler: StrategyScheduler | None = None,
) -> list[str]:
    scheduler = scheduler or StrategyScheduler(SchedulerConfig(mode="single", strategy=strategy))
    selected, _ = scheduler.select_batch(pool_ids, k, context or build_context(), state={})
    return selected


def mean_selected_score(
    selected_ids: Sequence[str],
    probabilities_by_id: dict[str, Sequence[float]],
    strategy: str,
) -> float:
    if not selected_ids:
        return 0.0
    scorer = score_function(strategy)
    rows = normalize_probability_rows([probabilities_by_id[sample_id] for sample_id in selected_ids], selected_ids)
    return sum(scorer(row) for row in rows) / len(rows)


def _elapsed_seconds(callback: Callable[[], Any], repeats: int) -> float:
    started = time.perf_counter()
    for _ in range(repeats):
        callback()
    return time.perf_counter() - started


def external_adapter_status() -> dict[str, str]:
    return {
        "modAL": (
            "not run: package importable but no benchmark adapter implemented"
            if importlib.util.find_spec("modAL") is not None
            else "not run: modAL is not importable"
        ),
        "skactiveml": (
            "not run: package importable but no benchmark adapter implemented"
            if importlib.util.find_spec("skactiveml") is not None
            else "not run: skactiveml is not importable"
        ),
    }


def compare_strategy(strategy: str, *, budget: int, repeats: int) -> dict[str, Any]:
    pool_ids = [candidate.sample_id for candidate in AUDIT_CANDIDATES]
    probabilities_by_id = probability_map()
    context = build_context()
    scheduler = StrategyScheduler(SchedulerConfig(mode="single", strategy=strategy))

    manual_selected = manual_select(pool_ids, budget, probabilities_by_id, strategy)
    sdk_selected = sdk_select(pool_ids, budget, strategy, context, scheduler)
    manual_elapsed = _elapsed_seconds(lambda: manual_select(pool_ids, budget, probabilities_by_id, strategy), repeats)
    sdk_elapsed = _elapsed_seconds(lambda: sdk_select(pool_ids, budget, strategy, context, scheduler), repeats)

    manual_set = set(manual_selected)
    sdk_set = set(sdk_selected)
    union = manual_set | sdk_set
    manual_mean_score = mean_selected_score(manual_selected, probabilities_by_id, strategy)
    sdk_mean_score = mean_selected_score(sdk_selected, probabilities_by_id, strategy)
    overhead_ratio = sdk_elapsed / manual_elapsed if manual_elapsed > 0 else None
    exact_order_match = manual_selected == sdk_selected

    return {
        "strategy": strategy,
        "budget": budget,
        "repeats": repeats,
        "candidate_count": len(pool_ids),
        "manual_selected_ids": manual_selected,
        "sdk_selected_ids": sdk_selected,
        "exact_order_match": exact_order_match,
        "overlap_count": len(manual_set & sdk_set),
        "jaccard": len(manual_set & sdk_set) / len(union) if union else 1.0,
        "manual_elapsed_seconds": manual_elapsed,
        "sdk_elapsed_seconds": sdk_elapsed,
        "overhead_ratio": overhead_ratio,
        "manual_mean_score": manual_mean_score,
        "sdk_mean_score": sdk_mean_score,
        "score_delta_sdk_minus_manual": sdk_mean_score - manual_mean_score,
        "sdk_worse_runtime": bool(overhead_ratio is not None and overhead_ratio > 1.0),
        "sdk_worse_quality": not exact_order_match or sdk_mean_score < manual_mean_score,
    }


def build_findings(rows: Sequence[dict[str, Any]], adapter_status: dict[str, str]) -> list[str]:
    findings: list[str] = []
    runtime_worse = [row for row in rows if row["sdk_worse_runtime"]]
    quality_worse = [row for row in rows if row["sdk_worse_quality"]]

    if runtime_worse:
        worst = max(runtime_worse, key=lambda row: float(row["overhead_ratio"] or 0.0))
        findings.append(
            "In this smoke run, SDK selection was slower than the direct manual formula loop; "
            f"largest measured overhead is {float(worst['overhead_ratio']):.2f}x for {worst['strategy']}."
        )
    else:
        findings.append("This run did not measure SDK selection as slower than the manual loop; rerun before relying on that result.")

    if quality_worse:
        findings.append(
            "At least one SDK selected order differs from the manual formula baseline, so formula parity needs investigation."
        )
    else:
        findings.append("SDK and manual formula baselines selected the same ordered batches for all audited strategies.")

    not_run = [name for name, status in adapter_status.items() if status.startswith("not run:")]
    if not_run:
        findings.append(
            "Direct external-library analog benchmarks were not run for "
            + ", ".join(not_run)
            + "; adapter status is reported separately and not simulated."
        )
    return findings


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: _csv_value(value) for key, value in row.items()} for row in rows])


def sanitize_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_json(payload), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def write_analysis(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# SDK vs Manual Audit Benchmark",
        "",
        "## Findings",
        "",
    ]
    lines.extend(f"- {finding}" for finding in summary["findings"])
    lines.extend(
        [
            "",
            "## Caveats",
            "",
        ]
    )
    lines.extend(f"- {caveat}" for caveat in summary["caveats"])
    lines.extend(
        [
            "",
            "## External Adapters",
            "",
        ]
    )
    lines.extend(f"- {name}: {status}" for name, status in sorted(summary["external_adapters"].items()))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _default_output_dir() -> Path:
    return REPO_ROOT / "benchmarks" / "results" / "audit_sdk_vs_manual" / time.strftime("%Y%m%d-%H%M%S")


def _ensure_output_dir_is_writable(output_dir: Path, *, overwrite: bool) -> None:
    output_files = [
        output_dir / "comparison.csv",
        output_dir / "summary.json",
        output_dir / "external_adapters.json",
        output_dir / "analysis.md",
    ]
    existing = [path for path in output_files if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing benchmark artifacts in {output_dir}: {names}. "
            "Use --overwrite for intentional replacement or choose a fresh output directory."
        )


def run_benchmark(output_dir: Path, *, budget: int = 5, repeats: int = 200, overwrite: bool = False) -> dict[str, Any]:
    if budget < 0:
        raise ValueError("budget must be non-negative.")
    if repeats <= 0:
        raise ValueError("repeats must be positive.")

    _ensure_output_dir_is_writable(output_dir, overwrite=overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [compare_strategy(strategy, budget=budget, repeats=repeats) for strategy in SUPPORTED_STRATEGIES]
    adapter_status = external_adapter_status()
    summary = {
        "run_id": time.strftime("%Y%m%d-%H%M%S"),
        "benchmark_contract": (
            "Compare direct in-memory manual uncertainty formulas with the SDK StrategyScheduler path on the same "
            "candidate ids, probabilities, label schema, budget, and model id. The SDK path is timed after warming an "
            "in-memory PredictionCache and scheduler capability cache, mirroring the manual path's precomputed "
            "probability map and steady-state formula loop. Manual tie-breaking is sample-id order; the fixture avoids "
            "score ties, while SDK production tie-breaking remains deterministic and model-id dependent."
        ),
        "workload": {
            "candidate_count": len(AUDIT_CANDIDATES),
            "labels": LABELS,
            "budget": budget,
            "repeats": repeats,
            "strategies": list(SUPPORTED_STRATEGIES),
        },
        "row_counts": {"comparison": len(rows)},
        "comparisons": rows,
        "external_adapters": adapter_status,
        "findings": build_findings(rows, adapter_status),
        "caveats": CAVEATS,
    }

    write_csv(output_dir / "comparison.csv", rows)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "external_adapters.json", adapter_status)
    write_analysis(output_dir / "analysis.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit SDK selection overhead against direct manual formulas.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    summary = run_benchmark(output_dir, budget=args.budget, repeats=args.repeats, overwrite=args.overwrite)
    print(f"Wrote SDK/manual audit artifacts to {output_dir.resolve()}")
    print(f"Comparison rows: {summary['row_counts']['comparison']}")
    for finding in summary["findings"]:
        print(f"- {finding}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
