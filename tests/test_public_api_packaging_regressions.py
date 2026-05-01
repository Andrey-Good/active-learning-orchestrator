from __future__ import annotations

import importlib.abc
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _project_optional_dependencies() -> dict[str, list[str]]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    return pyproject["project"]["optional-dependencies"]


def test_advertised_xxhash_extra_is_declared_in_package_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from active_learning_sdk import configs as configs_module
    from active_learning_sdk.configs import FingerprintConfig
    from active_learning_sdk.exceptions import ConfigurationError

    monkeypatch.setattr(configs_module, "_has_xxhash", lambda: False)

    with pytest.raises(ConfigurationError) as exc_info:
        FingerprintConfig(hash_algo="xxhash64").validate()

    match = re.search(r"active-learning-sdk\[([A-Za-z0-9_.-]+)\]", str(exc_info.value))
    assert match is not None, "The missing-xxhash error should point users at an installable extra."

    advertised_extra = match.group(1)
    optional_dependencies = _project_optional_dependencies()

    assert advertised_extra in optional_dependencies
    assert any(requirement.lower().startswith("xxhash") for requirement in optional_dependencies[advertised_extra])
    assert any(requirement.lower().startswith("xxhash") for requirement in optional_dependencies["all"])


def test_missing_sklearn_extra_reports_actionable_public_adapter_error() -> None:
    class BlockSklearn(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname: str, path: object = None, target: object = None) -> object:
            if fullname == "sklearn" or fullname.startswith("sklearn."):
                raise ModuleNotFoundError("blocked optional dependency: sklearn")
            return None

    for module_name in list(sys.modules):
        if (
            module_name == "active_learning_sdk.adapters"
            or module_name.startswith("active_learning_sdk.adapters.")
            or module_name == "sklearn"
            or module_name.startswith("sklearn.")
        ):
            sys.modules.pop(module_name, None)

    blocker = BlockSklearn()
    sys.meta_path.insert(0, blocker)
    try:
        import active_learning_sdk.adapters as adapters

        with pytest.raises((ImportError, RuntimeError)) as exc_info:
            getattr(adapters, "SklearnTextClassifierAdapter")
    finally:
        sys.meta_path.remove(blocker)
        for module_name in list(sys.modules):
            if module_name == "active_learning_sdk.adapters.sklearn" or module_name.startswith("sklearn."):
                sys.modules.pop(module_name, None)

    message = str(exc_info.value)
    assert "active-learning-sdk[sklearn]" in message
    assert "SklearnTextClassifierAdapter" in message


def test_root_public_api_exports_stable_runtime_contracts() -> None:
    import active_learning_sdk as sdk

    expected_exports = {
        "ActiveLearningProject",
        "SelectionContext",
        "StrategyScheduler",
        "StepResult",
        "DataSample",
        "AnnotationRecord",
        "ResolvedLabel",
        "MetricRecord",
        "SampleStatus",
        "RoundStatus",
        "StepKind",
        "LabelBackend",
        "RoundPushResult",
        "RoundProgress",
        "RoundPullResult",
        "TextClassificationAdapter",
        "ModelCapabilities",
        "inspect_model_capabilities",
        "CacheStore",
        "InMemoryCacheStore",
        "JsonlDiskCacheStore",
        "PredictionCache",
        "EmbeddingCache",
    }

    assert expected_exports <= set(sdk.__all__)
    for name in expected_exports:
        assert getattr(sdk, name) is not None


def test_built_artifacts_include_managed_docker_runtime_assets(tmp_path: Path) -> None:
    subprocess.run(
        ["uv", "build", "--wheel", "--sdist", "--out-dir", str(tmp_path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    wheel_path = next(tmp_path.glob("active_learning_sdk-*.whl"))
    sdist_path = next(tmp_path.glob("active_learning_sdk-*.tar.gz"))
    wheel_assets = {
        "active_learning_sdk/backends/assets/__init__.py",
        "active_learning_sdk/backends/assets/label_studio/docker-compose.yml",
        "active_learning_sdk/backends/assets/label_studio/nginx.conf",
    }

    with zipfile.ZipFile(wheel_path) as wheel:
        wheel_names = set(wheel.namelist())
    assert wheel_assets <= wheel_names
    assert not any(name.endswith((".pyc", ".pyo")) or "__pycache__/" in name for name in wheel_names)

    with tarfile.open(sdist_path) as sdist:
        sdist_names = set(sdist.getnames())
    sdist_suffixes = {name.split("/", 1)[1] for name in sdist_names if "/" in name}
    assert {
        "src/active_learning_sdk/backends/assets/__init__.py",
        "src/active_learning_sdk/backends/assets/label_studio/docker-compose.yml",
        "src/active_learning_sdk/backends/assets/label_studio/nginx.conf",
    } <= sdist_suffixes
    assert not any(name.endswith((".pyc", ".pyo")) or "__pycache__/" in name for name in sdist_names)
