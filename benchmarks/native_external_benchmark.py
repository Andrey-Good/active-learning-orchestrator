from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import sdk_first_benchmark as sdk_benchmark


CLAIM_CATEGORY = "native_external_library_workflow_smoke"
SMOKE_DATASET = "separable_topics"
SMOKE_SEED = 13
SMOKE_INITIAL_SEED_SIZE = 9
SMOKE_BATCH_SIZE = 4


@dataclass(frozen=True)
class NativeStrategySpec:
    name: str
    library: str
    native_api: str
    runner: Callable[["NativeSmokeContext", int], list[str]]


@dataclass(frozen=True)
class NativeSmokeContext:
    dataset: sdk_benchmark.BenchmarkDataset
    train_ids: list[str]
    test_ids: list[str]
    labeled_ids: list[str]
    pool_ids: list[str]
    pool_texts: list[str]
    x_pool: Any
    model: sdk_benchmark.SklearnTextBenchmarkAdapter
    sklearn_estimator: Any


class NativeBenchmarkSkip(RuntimeError):
    """Expected skip for missing optional libraries or incompatible native APIs."""


class ProbabilityEstimatorProxy:
    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def predict_proba(self, x_values: Any) -> Any:
        return self.estimator.predict_proba(x_values)


def _module_import_error(module_name: str, error: BaseException) -> str:
    return f"{module_name} is unavailable or incompatible: {error.__class__.__name__}: {error}"


def _import_optional_module(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except Exception as error:  # noqa: BLE001 - native optional imports must become skipped evidence.
        raise NativeBenchmarkSkip(_module_import_error(module_name, error)) from error


def _matrix_row_count(matrix: Any) -> int:
    shape = getattr(matrix, "shape", None)
    if shape is not None:
        return int(shape[0])
    return len(matrix)


def _extract_indices(raw_result: Any) -> list[int]:
    result = raw_result
    if isinstance(result, tuple):
        if not result:
            return []
        result = result[0]

    if hasattr(result, "tolist"):
        result = result.tolist()
    if isinstance(result, slice):
        return list(range(result.start or 0, result.stop or 0, result.step or 1))
    if isinstance(result, int):
        return [result]

    indices: list[int] = []
    for value in list(result):
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bool):
            raise NativeBenchmarkSkip("native API returned boolean values instead of integer indices")
        try:
            indices.append(int(value))
        except (TypeError, ValueError) as error:
            raise NativeBenchmarkSkip(f"native API returned a non-integer index: {value!r}") from error
    return indices


def _ids_from_pool_indices(pool_ids: Sequence[str], raw_result: Any, batch_size: int) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for index in _extract_indices(raw_result):
        if index < 0 or index >= len(pool_ids):
            raise NativeBenchmarkSkip(f"native API returned out-of-range pool index {index}")
        sample_id = pool_ids[index]
        if sample_id not in seen:
            selected.append(sample_id)
            seen.add(sample_id)
        if len(selected) >= batch_size:
            break
    return selected


def _run_modal_uncertainty(function_name: str, context: NativeSmokeContext, batch_size: int) -> list[str]:
    uncertainty_module = _import_optional_module("modAL.uncertainty")
    sampling_function = getattr(uncertainty_module, function_name, None)
    if sampling_function is None:
        raise NativeBenchmarkSkip(f"modAL.uncertainty.{function_name} is not available")

    try:
        raw_result = sampling_function(
            ProbabilityEstimatorProxy(context.sklearn_estimator),
            context.x_pool,
            n_instances=batch_size,
        )
    except Exception as error:  # noqa: BLE001 - incompatible native APIs should skip a row, not fail the run.
        raise NativeBenchmarkSkip(f"modAL.uncertainty.{function_name} failed: {error.__class__.__name__}: {error}") from error
    return _ids_from_pool_indices(context.pool_ids, raw_result, batch_size)


def _run_modal_entropy(context: NativeSmokeContext, batch_size: int) -> list[str]:
    return _run_modal_uncertainty("entropy_sampling", context, batch_size)


def _run_modal_margin(context: NativeSmokeContext, batch_size: int) -> list[str]:
    return _run_modal_uncertainty("margin_sampling", context, batch_size)


def _run_modal_least_confidence(context: NativeSmokeContext, batch_size: int) -> list[str]:
    return _run_modal_uncertainty("uncertainty_sampling", context, batch_size)


def _run_skactiveml_uncertainty(context: NativeSmokeContext, batch_size: int) -> list[str]:
    pool_module = _import_optional_module("skactiveml.pool")
    uncertainty_sampling = getattr(pool_module, "UncertaintySampling", None)
    if uncertainty_sampling is None:
        raise NativeBenchmarkSkip("skactiveml.pool.UncertaintySampling is not available")

    missing_label: Any = None
    try:
        utils_module = _import_optional_module("skactiveml.utils")
        missing_label = getattr(utils_module, "MISSING_LABEL", None)
    except NativeBenchmarkSkip:
        missing_label = None

    y_pool = [missing_label for _ in range(_matrix_row_count(context.x_pool))]
    try:
        query_strategy = uncertainty_sampling(method="entropy")
    except TypeError:
        query_strategy = uncertainty_sampling()
    try:
        raw_result = query_strategy.query(
            X=context.x_pool,
            y=y_pool,
            clf=ProbabilityEstimatorProxy(context.sklearn_estimator),
            batch_size=batch_size,
            return_utilities=False,
        )
    except TypeError:
        try:
            raw_result = query_strategy.query(
                X=context.x_pool,
                y=y_pool,
                clf=ProbabilityEstimatorProxy(context.sklearn_estimator),
                batch_size=batch_size,
            )
        except Exception as error:  # noqa: BLE001
            raise NativeBenchmarkSkip(
                f"skactiveml.pool.UncertaintySampling.query failed: {error.__class__.__name__}: {error}"
            ) from error
    except Exception as error:  # noqa: BLE001
        raise NativeBenchmarkSkip(
            f"skactiveml.pool.UncertaintySampling.query failed: {error.__class__.__name__}: {error}"
        ) from error
    return _ids_from_pool_indices(context.pool_ids, raw_result, batch_size)


def native_strategy_specs() -> dict[str, NativeStrategySpec]:
    specs = [
        NativeStrategySpec(
            name="modal_native_entropy",
            library="modal",
            native_api="modAL.uncertainty.entropy_sampling",
            runner=_run_modal_entropy,
        ),
        NativeStrategySpec(
            name="modal_native_margin",
            library="modal",
            native_api="modAL.uncertainty.margin_sampling",
            runner=_run_modal_margin,
        ),
        NativeStrategySpec(
            name="modal_native_least_confidence",
            library="modal",
            native_api="modAL.uncertainty.uncertainty_sampling",
            runner=_run_modal_least_confidence,
        ),
        NativeStrategySpec(
            name="skactiveml_native_uncertainty",
            library="skactiveml",
            native_api="skactiveml.pool.UncertaintySampling.query",
            runner=_run_skactiveml_uncertainty,
        ),
    ]
    return {spec.name: spec for spec in specs}


def build_smoke_context() -> NativeSmokeContext:
    sdk_benchmark.ensure_benchmark_dependencies()
    dataset = sdk_benchmark.build_benchmark_dataset(SMOKE_DATASET, SMOKE_SEED)
    sample_by_id = {sample.sample_id: sample for sample in dataset.samples}
    train_ids = sorted(sample.sample_id for sample in dataset.samples if sample.split == "train")
    test_ids = sorted(sample.sample_id for sample in dataset.samples if sample.split == "test")
    labeled_ids = sdk_benchmark.choose_initial_seed(dataset, train_ids, SMOKE_INITIAL_SEED_SIZE, SMOKE_SEED)
    pool_ids = [sample_id for sample_id in train_ids if sample_id not in set(labeled_ids)]

    model = sdk_benchmark.SklearnTextBenchmarkAdapter(dataset.labels, SMOKE_SEED)
    sdk_benchmark.train_and_evaluate(model, dataset, labeled_ids, test_ids)
    if model.pipeline is None:
        raise RuntimeError("Smoke model did not produce a fitted sklearn pipeline.")
    vectorizer = model.pipeline.named_steps["tfidf"]
    estimator = model.pipeline.named_steps["clf"]
    pool_texts = [sample_by_id[sample_id].text for sample_id in pool_ids]

    return NativeSmokeContext(
        dataset=dataset,
        train_ids=train_ids,
        test_ids=test_ids,
        labeled_ids=labeled_ids,
        pool_ids=pool_ids,
        pool_texts=pool_texts,
        x_pool=vectorizer.transform(pool_texts),
        model=model,
        sklearn_estimator=estimator,
    )


def _label_counts(dataset: sdk_benchmark.BenchmarkDataset, sample_ids: Sequence[str]) -> dict[str, int]:
    counts = {label: 0 for label in dataset.labels}
    sample_by_id = {sample.sample_id: sample for sample in dataset.samples}
    for sample_id in sample_ids:
        counts[sample_by_id[sample_id].label] = counts.get(sample_by_id[sample_id].label, 0) + 1
    return {label: count for label, count in sorted(counts.items()) if count}


def _evaluate_after_selection(context: NativeSmokeContext, selected_ids: Sequence[str]) -> dict[str, float]:
    model = sdk_benchmark.SklearnTextBenchmarkAdapter(context.dataset.labels, SMOKE_SEED)
    return sdk_benchmark.train_and_evaluate(
        model,
        context.dataset,
        [*context.labeled_ids, *selected_ids],
        context.test_ids,
    )


def skipped_result_row(spec: NativeStrategySpec, reason: str, elapsed_seconds: float) -> dict[str, Any]:
    return {
        "claim_category": CLAIM_CATEGORY,
        "dataset": SMOKE_DATASET,
        "seed": SMOKE_SEED,
        "library": spec.library,
        "strategy": spec.name,
        "native_api": spec.native_api,
        "status": "skipped",
        "skip_reason": reason,
        "requested_batch_size": SMOKE_BATCH_SIZE,
        "selected_count": 0,
        "selected_ids_json": "[]",
        "selected_label_counts_json": "{}",
        "macro_f1_after_selection": None,
        "accuracy_after_selection": None,
        "runtime_seconds": elapsed_seconds,
    }


def run_strategy(context: NativeSmokeContext, spec: NativeStrategySpec) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        selected_ids = spec.runner(context, min(SMOKE_BATCH_SIZE, len(context.pool_ids)))
        if not selected_ids:
            raise NativeBenchmarkSkip("native API returned no selected indices")
        metrics = _evaluate_after_selection(context, selected_ids)
    except NativeBenchmarkSkip as error:
        return skipped_result_row(spec, str(error), time.perf_counter() - started)

    return {
        "claim_category": CLAIM_CATEGORY,
        "dataset": context.dataset.name,
        "seed": SMOKE_SEED,
        "library": spec.library,
        "strategy": spec.name,
        "native_api": spec.native_api,
        "status": "ok",
        "skip_reason": "",
        "requested_batch_size": SMOKE_BATCH_SIZE,
        "selected_count": len(selected_ids),
        "selected_ids_json": json.dumps(selected_ids, sort_keys=True),
        "selected_label_counts_json": json.dumps(_label_counts(context.dataset, selected_ids), sort_keys=True),
        "macro_f1_after_selection": metrics["macro_f1"],
        "accuracy_after_selection": metrics["accuracy"],
        "runtime_seconds": time.perf_counter() - started,
    }


def parse_csv_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an opt-in native external active-learning query smoke. "
            "This does not run by default and does not require external active-learning libraries."
        )
    )
    parser.add_argument("--preset", choices=["smoke"], default="smoke")
    parser.add_argument("--libraries", default="modal,skactiveml", help="Comma-separated libraries: modal,skactiveml.")
    parser.add_argument("--strategies", default=None, help="Comma-separated native strategy names.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty benchmark output directory.")
    return parser


def write_summary(output_dir: Path, rows: Sequence[dict[str, Any]], manifest: dict[str, Any]) -> None:
    status_counts = {
        "ok": sum(1 for row in rows if row["status"] == "ok"),
        "skipped": sum(1 for row in rows if row["status"] == "skipped"),
    }
    sdk_benchmark.write_strict_json(
        output_dir / "native_external_summary.json",
        {
            "claim_category": CLAIM_CATEGORY,
            "manifest": manifest,
            "row_counts": {"results": len(rows), **status_counts},
            "strategies": list(manifest["strategies"]),
            "skips": {
                row["strategy"]: row["skip_reason"]
                for row in rows
                if row["status"] == "skipped"
            },
        },
    )
    lines = [
        "# Native External Active-Learning Smoke Summary",
        "",
        f"- Claim category: `{CLAIM_CATEGORY}`",
        f"- Dataset: `{SMOKE_DATASET}`",
        f"- Rows: {len(rows)}",
        f"- OK rows: {status_counts['ok']}",
        f"- Skipped rows: {status_counts['skipped']}",
        "",
        "This is a native query API smoke, not a full end-to-end production comparison or performance-superiority claim.",
        "",
        "| Strategy | Library | Status | Selected | Skip reason |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        reason = str(row["skip_reason"]).replace("|", "\\|")
        lines.append(
            f"| {row['strategy']} | {row['library']} | {row['status']} | {row['selected_count']} | {reason} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    run_id = sdk_benchmark.make_run_id()
    specs = native_strategy_specs()
    libraries = parse_csv_list(args.libraries)
    unknown_libraries = sorted(set(libraries) - {"modal", "skactiveml"})
    if unknown_libraries:
        raise SystemExit(f"Unknown libraries: {', '.join(unknown_libraries)}")

    strategies = parse_csv_list(args.strategies) if args.strategies else list(specs)
    unknown_strategies = sorted(set(strategies) - set(specs))
    if unknown_strategies:
        raise SystemExit(f"Unknown strategies: {', '.join(unknown_strategies)}")
    selected_specs = [specs[name] for name in strategies if specs[name].library in set(libraries)]
    if not selected_specs:
        raise SystemExit("No native external strategies selected after --libraries/--strategies filters.")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else sdk_benchmark.default_output_dir_for_preset("native_external_smoke", run_id)
    )
    sdk_benchmark.prepare_output_dir(output_dir, overwrite=args.overwrite)
    started = time.perf_counter()
    context = build_smoke_context()
    rows = [run_strategy(context, spec) for spec in selected_specs]

    manifest = {
        **sdk_benchmark.collect_reproducibility_metadata(raw_argv),
        "run_id": run_id,
        "preset": args.preset,
        "claim_category": CLAIM_CATEGORY,
        "benchmark_claim_category": CLAIM_CATEGORY,
        "datasets": [SMOKE_DATASET],
        "libraries": libraries,
        "strategies": [spec.name for spec in selected_specs],
        "seed": SMOKE_SEED,
        "initial_seed_size": SMOKE_INITIAL_SEED_SIZE,
        "batch_size": SMOKE_BATCH_SIZE,
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            "results_csv": "native_external_results.csv",
            "summary_json": "native_external_summary.json",
            "manifest_json": "manifest.json",
            "summary_md": "summary.md",
        },
        "artifact_names": [
            "native_external_results.csv",
            "native_external_summary.json",
            "manifest.json",
            "summary.md",
        ],
        "benchmark_contract": (
            "Opt-in native-query smoke for optional external active-learning APIs. "
            "Rows call native query functions/classes when importable. Missing or incompatible optional libraries "
            "produce skipped rows with reasons. This artifact category is not a full external-library benchmark "
            "and must not be used as a performance-superiority claim."
        ),
    }

    sdk_benchmark.write_csv(output_dir / "native_external_results.csv", rows)
    sdk_benchmark.write_strict_json(output_dir / "manifest.json", manifest)
    write_summary(output_dir, rows, manifest)

    print(f"Wrote native external benchmark artifacts to {output_dir.resolve()}")
    print(f"Result rows: {len(rows)}")
    print(f"OK rows: {sum(1 for row in rows if row['status'] == 'ok')}")
    print(f"Skipped rows: {sum(1 for row in rows if row['status'] == 'skipped')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
