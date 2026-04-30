from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACTS_PATH = REPO_ROOT / "docs" / "SDK_CONTRACTS.md"


def _documented_stable_exports() -> list[str]:
    contracts = CONTRACTS_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"<!-- PUBLIC_CONTRACT_STABLE_EXPORTS\n(?P<exports>.*?)\n"
        r"END_PUBLIC_CONTRACT_STABLE_EXPORTS -->",
        contracts,
        flags=re.DOTALL,
    )
    assert match is not None, "SDK_CONTRACTS.md must document stable root exports."
    return [
        line.strip()
        for line in match.group("exports").splitlines()
        if line.strip()
    ]


def _clear_sdk_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "active_learning_sdk" or module_name.startswith(
            "active_learning_sdk."
        ):
            sys.modules.pop(module_name, None)


def test_root_all_matches_documented_stable_exports() -> None:
    _clear_sdk_modules()
    sdk = importlib.import_module("active_learning_sdk")

    documented_exports = _documented_stable_exports()

    assert list(sdk.__all__) == documented_exports


def test_documented_stable_exports_are_importable_from_root() -> None:
    _clear_sdk_modules()
    sdk = importlib.import_module("active_learning_sdk")

    for export_name in _documented_stable_exports():
        assert getattr(sdk, export_name) is not None


def test_root_import_does_not_require_optional_concrete_adapters() -> None:
    _clear_sdk_modules()

    importlib.import_module("active_learning_sdk")

    assert "active_learning_sdk.adapters.sklearn" not in sys.modules
    assert "active_learning_sdk.adapters.huggingface" not in sys.modules


def test_public_exceptions_are_importable_and_subclass_base_error() -> None:
    _clear_sdk_modules()
    sdk = importlib.import_module("active_learning_sdk")

    exception_exports = [
        "ConfigurationError",
        "DatasetMismatchError",
        "ProjectLockedError",
        "StateCorruptedError",
        "InfrastructureError",
        "LabelBackendError",
        "StrategyError",
        "ModelAdapterError",
        "StopCriteriaReached",
    ]

    assert issubclass(sdk.ActiveLearningError, RuntimeError)
    for export_name in exception_exports:
        assert issubclass(getattr(sdk, export_name), sdk.ActiveLearningError)
