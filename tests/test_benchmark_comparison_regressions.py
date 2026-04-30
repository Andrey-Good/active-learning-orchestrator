from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import manual_strategy_benchmark


def test_manual_and_sdk_select_same_batches_on_frozen_probabilities() -> None:
    pool_ids = [candidate.sample_id for candidate in manual_strategy_benchmark.AUDIT_CANDIDATES]
    probabilities_by_id = manual_strategy_benchmark.probability_map()
    context = manual_strategy_benchmark.build_context()

    for strategy in manual_strategy_benchmark.SUPPORTED_STRATEGIES:
        manual_selected = manual_strategy_benchmark.manual_select(
            pool_ids, 5, probabilities_by_id, strategy
        )
        sdk_selected = manual_strategy_benchmark.sdk_select(pool_ids, 5, strategy, context)

        assert sdk_selected == manual_selected
        assert len(sdk_selected) == 5
        assert len(set(sdk_selected)) == 5


@pytest.mark.parametrize(
    ("probabilities", "match"),
    [
        ([[5.0, 5.0]], "sum to 1.0"),
        ([[0.6, 0.6]], "sum to 1.0"),
        ([[2.0, -1.0]], "non-negative"),
        ([[True, False]], "numeric"),
        ([[1.0]], "at least 2 columns"),
        ([[0.5, 0.5], [0.25, 0.25, 0.5]], "width"),
    ],
)
def test_audit_probability_rows_use_strict_sdk_contract(probabilities: object, match: str) -> None:
    pool_ids = [f"s{index}" for index, _ in enumerate(probabilities, start=1)]

    with pytest.raises(ValueError, match=match):
        manual_strategy_benchmark.normalize_probability_rows(probabilities, pool_ids)


def test_run_benchmark_writes_stable_audit_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "audit"

    summary = manual_strategy_benchmark.run_benchmark(output_dir, budget=4, repeats=3)

    assert summary["workload"]["candidate_count"] == len(
        manual_strategy_benchmark.AUDIT_CANDIDATES
    )
    assert summary["workload"]["budget"] == 4
    assert summary["row_counts"] == {
        "comparison": len(manual_strategy_benchmark.SUPPORTED_STRATEGIES)
    }
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "comparison.csv").is_file()
    assert (output_dir / "external_adapters.json").is_file()
    assert (output_dir / "analysis.md").is_file()

    saved_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["benchmark_contract"]
    assert saved_summary["caveats"]
    assert saved_summary["findings"]
    assert all(isinstance(row["manual_selected_ids"], list) for row in saved_summary["comparisons"])
    assert all(isinstance(row["sdk_selected_ids"], list) for row in saved_summary["comparisons"])

    rows = list(csv.DictReader((output_dir / "comparison.csv").open(encoding="utf-8")))
    assert {row["strategy"] for row in rows} == set(
        manual_strategy_benchmark.SUPPORTED_STRATEGIES
    )
    assert all(row["exact_order_match"] == "True" for row in rows)
    assert all(float(row["jaccard"]) == 1.0 for row in rows)
    assert all(float(row["manual_elapsed_seconds"]) >= 0.0 for row in rows)
    assert all(float(row["sdk_elapsed_seconds"]) >= 0.0 for row in rows)


def test_run_benchmark_refuses_to_overwrite_existing_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "audit"

    manual_strategy_benchmark.run_benchmark(output_dir, budget=2, repeats=1)

    try:
        manual_strategy_benchmark.run_benchmark(output_dir, budget=2, repeats=1)
    except FileExistsError as exc:
        assert "--overwrite" in str(exc)
    else:
        raise AssertionError("run_benchmark should reject accidental artifact overwrite")

    summary = manual_strategy_benchmark.run_benchmark(
        output_dir, budget=2, repeats=1, overwrite=True
    )
    assert summary["workload"]["budget"] == 2


def test_external_adapter_status_never_fabricates_unavailable_comparisons() -> None:
    statuses = manual_strategy_benchmark.external_adapter_status()

    assert set(statuses) == {"modAL", "skactiveml"}
    assert all(status.startswith("not run:") for status in statuses.values())


def test_cli_entrypoint_returns_success_and_uses_requested_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli"

    result = manual_strategy_benchmark.main(
        ["--output-dir", str(output_dir), "--budget", "3", "--repeats", "2"]
    )

    assert result == 0
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["workload"]["budget"] == 3
    assert summary["workload"]["repeats"] == 2
