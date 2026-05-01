from __future__ import annotations

import json
import sys
import csv
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import reference_strategy_benchmark
from benchmarks import sdk_first_benchmark


def _load_manifest(output_dir: Path) -> dict[str, Any]:
    return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))


def _assert_reproducibility_metadata(manifest: dict[str, Any], argv: list[str]) -> None:
    assert manifest["artifact_schema_version"] == sdk_first_benchmark.BENCHMARK_ARTIFACT_SCHEMA_VERSION
    assert manifest["argv"] == argv

    assert set(["sha", "dirty", "status_entry_count", "status_count"]).issubset(manifest["git"])
    assert manifest["git"]["status_entry_count"] == manifest["git"]["status_count"]

    runtime = manifest["runtime"]
    assert runtime["python_version"]
    assert runtime["python_version_info"]["major"] == sys.version_info.major
    assert runtime["platform"]
    assert runtime["platform_system"]
    assert runtime["platform_machine"] is not None


def _assert_artifact_metadata(manifest: dict[str, Any]) -> None:
    artifact_values = sorted(set(manifest["artifacts"].values()))
    assert manifest["artifact_names"] == artifact_values
    assert "manifest.json" in artifact_values


def test_sdk_first_smoke_manifest_records_evidence_contract(tmp_path: Path) -> None:
    output_dir = tmp_path / "sdk_first"
    argv = [
        "--preset",
        "smoke",
        "--datasets",
        "separable_topics",
        "--strategies",
        "random",
        "--budgets",
        "12",
        "--seeds",
        "13",
        "--output-dir",
        str(output_dir),
    ]

    assert sdk_first_benchmark.main(argv) == 0

    manifest = _load_manifest(output_dir)
    _assert_reproducibility_metadata(manifest, argv)
    _assert_artifact_metadata(manifest)
    assert manifest["benchmark_claim_category"] == sdk_first_benchmark.CLAIM_CATEGORY_ACTIVE_LEARNING_QUALITY
    assert "StrategyScheduler" in manifest["benchmark_contract"]
    assert "not native external-library workflow evidence" in manifest["benchmark_contract"]
    assert "not true MC-dropout" in manifest["benchmark_contract"]
    assert "independently trained committee quality evidence" in manifest["benchmark_contract"]
    assert "integration and diagnostics only" in manifest["strategy_evidence_notes"]["stochastic_committee_proxy"]
    assert manifest["artifacts"]["metrics_csv"] == "metrics.csv"
    assert manifest["artifacts"]["validation_json"] == "validation.json"


def test_sdk_first_smoke_records_budgets_below_initial_seed_as_warnings(tmp_path: Path) -> None:
    output_dir = tmp_path / "sdk_first_budget_warnings"
    argv = [
        "--preset",
        "smoke",
        "--datasets",
        "separable_topics",
        "--strategies",
        "random",
        "--budgets",
        "4,12",
        "--seeds",
        "13",
        "--output-dir",
        str(output_dir),
    ]

    assert sdk_first_benchmark.main(argv) == 0

    manifest = _load_manifest(output_dir)
    warning_rows = list(csv.DictReader((output_dir / "budget_warnings.csv").open(encoding="utf-8")))
    assert manifest["artifacts"]["budget_warnings_csv"] == "budget_warnings.csv"
    assert manifest["skipped_budgets"][0]["requested_budget"] == 4
    assert warning_rows[0]["requested_budget"] == "4"
    assert warning_rows[0]["reason"] == "below_initial_seed_size"
    assert warning_rows[0]["nearest_executable_budget"] == "9"


def test_reference_smoke_manifest_records_formula_parity_contract(tmp_path: Path) -> None:
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

    assert reference_strategy_benchmark.main(argv) == 0

    manifest = _load_manifest(output_dir)
    _assert_reproducibility_metadata(manifest, argv)
    _assert_artifact_metadata(manifest)
    assert manifest["benchmark_claim_category"] == reference_strategy_benchmark.CLAIM_CATEGORY_FORMULA_PARITY
    assert "rows are not renormalized" in manifest["benchmark_contract"]
    assert "does not call external scorer or query APIs" in manifest["benchmark_contract"]
    assert manifest["artifacts"]["equivalence_csv"] == "equivalence.csv"
    assert manifest["artifacts"]["external_adapters_json"] == "external_adapters.json"


def test_project_smoke_manifest_records_public_workflow_contract(tmp_path: Path) -> None:
    output_dir = tmp_path / "project_smoke"
    argv = [
        "--preset",
        "project_smoke",
        "--datasets",
        "separable_topics",
        "--strategies",
        "entropy",
        "--seeds",
        "13",
        "--output-dir",
        str(output_dir),
    ]

    sdk_first_benchmark.ensure_benchmark_dependencies()
    sdk_first_benchmark.run_project_smoke(
        output_dir=output_dir,
        seed=13,
        initial_seed_size=9,
        batch_size=6,
        argv=argv,
    )

    manifest = _load_manifest(output_dir)
    _assert_reproducibility_metadata(manifest, argv)
    _assert_artifact_metadata(manifest)
    assert manifest["benchmark_claim_category"] == (
        sdk_first_benchmark.CLAIM_CATEGORY_END_TO_END_PUBLIC_PROJECT_WORKFLOW
    )
    assert "public ActiveLearningProject facade" in manifest["benchmark_contract"]
    assert "does not measure external human-labeling services" in manifest["benchmark_contract"]
    assert manifest["artifacts"]["workdir_state"] == "workdir/state.json"
