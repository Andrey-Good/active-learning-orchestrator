from __future__ import annotations

import importlib
import sys


def test_adapters_package_does_not_import_optional_sklearn_adapter_eagerly() -> None:
    sys.modules.pop("active_learning_sdk.adapters", None)
    sys.modules.pop("active_learning_sdk.adapters.sklearn", None)

    adapters = importlib.import_module("active_learning_sdk.adapters")

    assert "SklearnTextClassifierAdapter" in adapters.__all__
    assert "active_learning_sdk.adapters.sklearn" not in sys.modules


def test_optional_adapter_export_is_loaded_lazily_on_attribute_access() -> None:
    sys.modules.pop("active_learning_sdk.adapters", None)
    sys.modules.pop("active_learning_sdk.adapters.sklearn", None)

    adapters = importlib.import_module("active_learning_sdk.adapters")
    adapter_class = adapters.SklearnTextClassifierAdapter

    assert adapter_class.__name__ == "SklearnTextClassifierAdapter"
    assert "active_learning_sdk.adapters.sklearn" in sys.modules
