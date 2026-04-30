from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Sequence

import numpy as np

from ..exceptions import ModelAdapterError
from .base import normalize_adapter_batch_size


_PROBABILITY_SUM_REL_TOL = 1e-9
_PROBABILITY_SUM_ABS_TOL = 1e-12
_MAX_SERIALIZED_ARRAY_ELEMENTS = 10_000
_FITTED_STATE_ATTRIBUTES = (
    "classes_",
    "coef_",
    "intercept_",
    "vocabulary_",
    "idf_",
    "feature_names_in_",
    "n_features_in_",
    "n_iter_",
    "components_",
    "mean_",
    "scale_",
    "var_",
    "class_log_prior_",
    "feature_log_prob_",
    "class_count_",
    "feature_count_",
)


def _sklearn_import_error() -> ImportError:
    return ImportError(
        "active_learning_sdk.adapters.sklearn requires the optional scikit-learn dependency. "
        "Install it with active-learning-sdk[sklearn]."
    )


def _default_sklearn_components() -> tuple[Any, Any, Any]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except ModuleNotFoundError as error:
        if error.name == "sklearn" or (error.name or "").startswith("sklearn."):
            raise _sklearn_import_error() from error
        raise
    return Pipeline, TfidfVectorizer, LogisticRegression


def _sklearn_metrics() -> tuple[Any, Any]:
    try:
        from sklearn.metrics import accuracy_score, f1_score
    except ModuleNotFoundError as error:
        if error.name == "sklearn" or (error.name or "").startswith("sklearn."):
            raise _sklearn_import_error() from error
        raise
    return accuracy_score, f1_score


def _check_is_fitted(estimator: Any) -> None:
    try:
        from sklearn.utils.validation import check_is_fitted
    except ModuleNotFoundError as error:
        if error.name == "sklearn" or (error.name or "").startswith("sklearn."):
            raise _sklearn_import_error() from error
        raise
    check_is_fitted(estimator)


class SklearnTextClassifierAdapter:
    """
    scikit-learn text-classification adapter.

    The default estimator is a fast deterministic TF-IDF + LogisticRegression
    pipeline suitable for tiny smoke-test datasets. Users may also inject any
    fitted-compatible sklearn estimator or Pipeline that accepts raw text.
    """

    def __init__(self, estimator: Any | None = None) -> None:
        self.estimator = estimator if estimator is not None else self._default_estimator()
        self._version = 0

    @staticmethod
    def _default_estimator() -> Any:
        Pipeline, TfidfVectorizer, LogisticRegression = _default_sklearn_components()
        return Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(min_df=1, ngram_range=(1, 2))),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=200,
                        random_state=0,
                        solver="liblinear",
                    ),
                ),
            ]
        )

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs: Any) -> None:
        text_list = list(texts)
        label_list = list(labels)
        if len(text_list) != len(label_list):
            raise ModelAdapterError(
                f"fit() received {len(text_list)} texts but {len(label_list)} labels; lengths must match."
            )

        try:
            self.estimator.fit(text_list, label_list, **kwargs)
        except Exception as error:
            raise ModelAdapterError(f"sklearn estimator fit() failed: {error}") from error
        self._classes()

        self._version += 1

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> List[List[float]]:
        text_list = list(texts)
        if not text_list:
            return []

        self._ensure_fitted()
        size = self._normalized_batch_size(batch_size)
        output: List[List[float]] = []
        for offset in range(0, len(text_list), size):
            batch = text_list[offset : offset + size]
            output.extend(self._predict_proba_batch(batch))
        if len(output) != len(text_list):
            raise ModelAdapterError(
                f"predict_proba() returned {len(output)} probability rows for {len(text_list)} input texts; "
                "counts must match."
            )
        return output

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> Dict[str, float]:
        text_list = list(texts)
        label_list = list(labels)
        if len(text_list) != len(label_list):
            raise ModelAdapterError(
                f"evaluate() received {len(text_list)} texts but {len(label_list)} labels; lengths must match."
            )
        if not text_list:
            return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}

        probabilities = self.predict_proba(text_list)
        classes = self._classes()
        predicted = [classes[int(np.argmax(row))] for row in probabilities]
        accuracy_score, f1_score = _sklearn_metrics()

        return {
            "accuracy": float(accuracy_score(label_list, predicted)),
            "macro_f1": float(f1_score(label_list, predicted, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(label_list, predicted, average="weighted", zero_division=0)),
        }

    def get_model_id(self) -> str:
        fingerprint = hashlib.sha256(self._estimator_config().encode("utf-8")).hexdigest()[:16]
        estimator_class = self._class_path(self.estimator)
        return f"{estimator_class}:v{self._version}:{fingerprint}"

    def _predict_proba_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if callable(getattr(self.estimator, "predict_proba", None)):
            try:
                probabilities = self.estimator.predict_proba(list(texts))
            except Exception as error:
                raise ModelAdapterError(f"sklearn estimator predict_proba() failed: {error}") from error
            return self._validate_probability_rows(probabilities, expected_row_count=len(texts))

        if callable(getattr(self.estimator, "decision_function", None)):
            try:
                scores = self.estimator.decision_function(list(texts))
            except Exception as error:
                raise ModelAdapterError(f"sklearn estimator decision_function() failed: {error}") from error
            return self._probabilities_from_decision_function(scores, expected_row_count=len(texts))

        raise ModelAdapterError(
            "sklearn estimator must provide predict_proba() or decision_function() to produce probabilities."
        )

    def _probabilities_from_decision_function(
        self, scores: Any, expected_row_count: int | None = None
    ) -> List[List[float]]:
        score_array = np.asarray(scores, dtype=float)
        classes = self._classes()

        if score_array.ndim == 1:
            if len(classes) != 2:
                raise ModelAdapterError(
                    "decision_function() returned one score per sample but the fitted estimator is not binary."
                )
            positive = 1.0 / (1.0 + np.exp(-score_array))
            probabilities = np.column_stack([1.0 - positive, positive])
            return self._validate_probability_rows(probabilities, expected_row_count=expected_row_count)

        if score_array.ndim == 2 and score_array.shape[1] == len(classes):
            shifted = score_array - np.max(score_array, axis=1, keepdims=True)
            exp_scores = np.exp(shifted)
            probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            return self._validate_probability_rows(probabilities, expected_row_count=expected_row_count)

        raise ModelAdapterError(
            "decision_function() output shape is incompatible with fitted estimator classes."
        )

    def _validate_probability_rows(
        self, probabilities: Any, expected_row_count: int | None = None
    ) -> List[List[float]]:
        try:
            probability_array = np.asarray(probabilities, dtype=float)
        except (TypeError, ValueError) as error:
            raise ModelAdapterError("predict_proba() must return a numeric probability matrix.") from error
        if probability_array.ndim != 2:
            raise ModelAdapterError("predict_proba() must return a 2D probability matrix.")
        if expected_row_count is not None and probability_array.shape[0] != expected_row_count:
            raise ModelAdapterError(
                f"predict_proba() returned {probability_array.shape[0]} probability rows for "
                f"{expected_row_count} input texts; counts must match."
            )
        if probability_array.shape[1] != len(self._classes()):
            raise ModelAdapterError(
                "predict_proba() column count does not match the fitted estimator class count."
            )

        rows: List[List[float]] = []
        for row in probability_array:
            values = [float(value) for value in row]
            if not all(math.isfinite(value) for value in values):
                raise ModelAdapterError("predict_proba() returned non-finite probabilities.")
            if any(value < 0 for value in values):
                raise ModelAdapterError("predict_proba() returned negative probabilities.")
            row_sum = sum(values)
            if row_sum <= 0:
                raise ModelAdapterError("predict_proba() returned a row with non-positive probability sum.")
            if not math.isclose(
                row_sum,
                1.0,
                rel_tol=_PROBABILITY_SUM_REL_TOL,
                abs_tol=_PROBABILITY_SUM_ABS_TOL,
            ):
                raise ModelAdapterError(
                    f"predict_proba() returned a probability row that must sum to 1.0; got {row_sum}."
                )
            rows.append(values)
        return rows

    def _classes(self) -> List[Any]:
        classes = getattr(self.estimator, "classes_", None)
        if classes is None:
            raise ModelAdapterError("sklearn estimator does not expose fitted classes_.")
        class_list = list(classes)
        if len(class_list) < 2:
            raise ModelAdapterError(
                "sklearn estimator must expose at least two fitted classes; "
                "one-class probability surfaces are not supported."
            )
        return class_list

    def _ensure_fitted(self) -> None:
        try:
            _check_is_fitted(self.estimator)
        except Exception as error:
            raise ModelAdapterError("sklearn estimator is not fitted.") from error

    @staticmethod
    def _normalized_batch_size(batch_size: int) -> int:
        return normalize_adapter_batch_size(batch_size)

    def _estimator_config(self) -> str:
        payload = self._fingerprint_value(self.estimator)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)

    @classmethod
    def _fingerprint_value(cls, value: Any, depth: int = 0) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            payload: Dict[str, Any] = {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
            if value.size > _MAX_SERIALIZED_ARRAY_ELEMENTS:
                if value.dtype.hasobject:
                    payload["array"] = {"omitted": "object-array-too-large"}
                else:
                    payload["array"] = {
                        "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                    }
            else:
                payload["array"] = value.tolist()
            return payload
        if hasattr(value, "shape") and all(
            hasattr(value, attribute) for attribute in ("data", "indices", "indptr")
        ):
            return {
                "class": cls._class_path(value),
                "data": cls._fingerprint_value(value.data, depth + 1),
                "indices": cls._fingerprint_value(value.indices, depth + 1),
                "indptr": cls._fingerprint_value(value.indptr, depth + 1),
                "shape": list(value.shape),
            }
        if hasattr(value, "toarray") and hasattr(value, "shape"):
            shape = getattr(value, "shape", None)
            if shape is not None and np.prod(shape, dtype=np.int64) > _MAX_SERIALIZED_ARRAY_ELEMENTS:
                return {
                    "class": cls._class_path(value),
                    "shape": list(shape),
                }
            try:
                dense = value.toarray()
            except Exception:
                return {
                    "class": cls._class_path(value),
                    "shape": list(shape) if shape is not None else None,
                }
            return cls._fingerprint_value(dense, depth + 1)
        if isinstance(value, (list, tuple)):
            return [cls._fingerprint_value(item, depth + 1) for item in value]
        if isinstance(value, dict):
            return {
                str(key): cls._fingerprint_value(val, depth + 1)
                for key, val in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, (set, frozenset)):
            return [cls._fingerprint_value(item, depth + 1) for item in sorted(value, key=str)]
        if hasattr(value, "get_params") and depth < 4:
            return {
                "class": cls._class_path(value),
                "params": cls._fingerprint_value(value.get_params(deep=False), depth + 1),
                "fitted_state": cls._fitted_state(value, depth + 1),
            }
        if hasattr(value, "shape"):
            return {
                "class": cls._class_path(value),
                "shape": list(value.shape),
            }
        return {"class": cls._class_path(value)}

    @staticmethod
    def _fitted_attribute_names(estimator: Any) -> List[str]:
        names = set(_FITTED_STATE_ATTRIBUTES)
        try:
            names.update(
                name
                for name in dir(estimator)
                if name.endswith("_") and not name.startswith("_") and not name.endswith("__")
            )
        except Exception:
            pass
        return sorted(names)

    @classmethod
    def _fitted_state(cls, estimator: Any, depth: int = 0) -> Any:
        if depth >= 4:
            return None

        state: Dict[str, Any] = {}
        for attribute in cls._fitted_attribute_names(estimator):
            try:
                value = getattr(estimator, attribute)
            except Exception:
                continue
            if callable(value):
                continue
            try:
                state[attribute] = cls._fingerprint_value(value, depth + 1)
            except Exception:
                shape = getattr(value, "shape", None)
                state[attribute] = {
                    "class": cls._class_path(value),
                    "shape": list(shape) if shape is not None else None,
                }

        steps = getattr(estimator, "steps", None)
        if steps:
            nested_steps: Dict[str, Any] = {}
            for name, step in steps:
                nested_steps[str(name)] = cls._fitted_state(step, depth + 1)
            state["steps"] = nested_steps

        return state or None

    @staticmethod
    def _class_path(value: Any) -> str:
        value_class = value.__class__
        return f"{value_class.__module__}.{value_class.__qualname__}"
