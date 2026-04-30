from __future__ import annotations

import builtins
import importlib
import sys
from typing import Any

import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from active_learning_sdk.adapters import SklearnTextClassifierAdapter
from active_learning_sdk.exceptions import ModelAdapterError


TEXTS = [
    "great fast useful product",
    "excellent helpful nice result",
    "bad slow broken product",
    "terrible useless poor result",
]
LABELS = ["positive", "positive", "negative", "negative"]


def test_default_adapter_fits_tiny_text_data_and_predicts_probabilities() -> None:
    adapter = SklearnTextClassifierAdapter()

    adapter.fit(TEXTS, LABELS)
    probabilities = adapter.predict_proba(["great helpful product", "bad broken result"])

    assert len(probabilities) == 2
    assert all(len(row) == 2 for row in probabilities)
    assert all(sum(row) == pytest.approx(1.0) for row in probabilities)


def test_evaluate_returns_required_metrics() -> None:
    adapter = SklearnTextClassifierAdapter()
    adapter.fit(TEXTS, LABELS)

    metrics = adapter.evaluate(TEXTS, LABELS)

    assert set(metrics) >= {"accuracy", "macro_f1", "weighted_f1"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["weighted_f1"] <= 1.0


def test_get_model_id_changes_after_successful_fit() -> None:
    adapter = SklearnTextClassifierAdapter()
    before = adapter.get_model_id()

    adapter.fit(TEXTS, LABELS)
    after = adapter.get_model_id()

    assert after != before


def test_get_model_id_distinguishes_same_config_differently_fitted_estimators() -> None:
    first = SklearnTextClassifierAdapter()
    second = SklearnTextClassifierAdapter()

    first.fit(TEXTS, LABELS)
    second.fit(
        [
            "alpha alpha alpha",
            "alpha beta alpha",
            "omega omega omega",
            "omega beta omega",
        ],
        LABELS,
    )

    assert first.get_model_id() != second.get_model_id()


def test_get_model_id_distinguishes_inverted_multinomial_nb_pipeline_labels() -> None:
    texts = [
        "alpha alpha signal",
        "alpha signal bright",
        "omega omega signal",
        "omega signal dark",
    ]
    labels = ["alpha", "alpha", "omega", "omega"]
    inverted_labels = ["omega", "omega", "alpha", "alpha"]

    first = SklearnTextClassifierAdapter(
        estimator=Pipeline(
            steps=[
                ("count", CountVectorizer()),
                ("classifier", MultinomialNB()),
            ]
        )
    )
    second = SklearnTextClassifierAdapter(
        estimator=Pipeline(
            steps=[
                ("count", CountVectorizer()),
                ("classifier", MultinomialNB()),
            ]
        )
    )

    first.fit(texts, labels)
    second.fit(texts, inverted_labels)

    assert first.get_model_id() != second.get_model_id()
    assert first.predict_proba(["alpha signal"]) != second.predict_proba(["alpha signal"])


def test_fit_rejects_mismatched_lengths() -> None:
    adapter = SklearnTextClassifierAdapter()

    with pytest.raises(ModelAdapterError, match="lengths must match"):
        adapter.fit(["one", "two"], ["positive"])


def test_adapter_works_with_injected_sklearn_pipeline() -> None:
    estimator = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(min_df=1)),
            ("classifier", LinearSVC(random_state=0)),
        ]
    )
    adapter = SklearnTextClassifierAdapter(estimator=estimator)

    adapter.fit(TEXTS, LABELS)
    probabilities = adapter.predict_proba(["excellent product", "terrible product"])

    assert len(probabilities) == 2
    assert all(len(row) == 2 for row in probabilities)
    assert all(sum(row) == pytest.approx(1.0) for row in probabilities)


class NoProbabilityEstimator(ClassifierMixin, BaseEstimator):
    classes_: list[str]

    def fit(self, texts: list[str], labels: list[Any]) -> "NoProbabilityEstimator":
        self.classes_ = sorted(set(labels))
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return [self.classes_[0] for _ in texts]


def test_estimator_without_probability_support_raises_clear_error() -> None:
    adapter = SklearnTextClassifierAdapter(estimator=NoProbabilityEstimator())
    adapter.fit(TEXTS, LABELS)

    with pytest.raises(ModelAdapterError, match="predict_proba\\(\\) or decision_function\\(\\)"):
        adapter.predict_proba(["sample text"])


class OneClassProbabilityEstimator(ClassifierMixin, BaseEstimator):
    classes_: list[str]

    def fit(self, texts: list[str], labels: list[Any]) -> "OneClassProbabilityEstimator":
        self.classes_ = sorted(set(labels))
        return self

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


def test_one_class_fitted_estimator_is_rejected() -> None:
    adapter = SklearnTextClassifierAdapter(estimator=OneClassProbabilityEstimator())

    with pytest.raises(ModelAdapterError, match="at least two fitted classes"):
        adapter.fit(["only positive"], ["positive"])


class ShortProbabilityEstimator(ClassifierMixin, BaseEstimator):
    classes_: list[str]

    def fit(self, texts: list[str], labels: list[Any]) -> "ShortProbabilityEstimator":
        self.classes_ = sorted(set(labels))
        return self

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        return [[0.25, 0.75] for _ in texts[:-1]]


def test_short_predict_proba_output_raises_model_adapter_error() -> None:
    adapter = SklearnTextClassifierAdapter(estimator=ShortProbabilityEstimator())
    adapter.fit(TEXTS, LABELS)

    with pytest.raises(ModelAdapterError, match="probability rows for 2 input texts"):
        adapter.predict_proba(["first text", "second text"])


def test_evaluate_wraps_short_predict_proba_output_as_model_adapter_error() -> None:
    adapter = SklearnTextClassifierAdapter(estimator=ShortProbabilityEstimator())
    adapter.fit(TEXTS, LABELS)

    with pytest.raises(ModelAdapterError, match="probability rows for 4 input texts"):
        adapter.evaluate(TEXTS, LABELS)


class NegativeProbabilityEstimator(ClassifierMixin, BaseEstimator):
    classes_: list[str]

    def fit(self, texts: list[str], labels: list[Any]) -> "NegativeProbabilityEstimator":
        self.classes_ = sorted(set(labels))
        return self

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        return [[-0.25, 1.25] for _ in texts]


def test_negative_probability_values_raise_model_adapter_error() -> None:
    adapter = SklearnTextClassifierAdapter(estimator=NegativeProbabilityEstimator())
    adapter.fit(TEXTS, LABELS)

    with pytest.raises(ModelAdapterError, match="negative probabilities"):
        adapter.predict_proba(["sample text"])


class CountLikeProbabilityEstimator(ClassifierMixin, BaseEstimator):
    classes_: list[str]

    def fit(self, texts: list[str], labels: list[Any]) -> "CountLikeProbabilityEstimator":
        self.classes_ = sorted(set(labels))
        return self

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        return [[5.0, 5.0] for _ in texts]


def test_count_like_predict_proba_rows_raise_model_adapter_error() -> None:
    adapter = SklearnTextClassifierAdapter(estimator=CountLikeProbabilityEstimator())
    adapter.fit(TEXTS, LABELS)

    with pytest.raises(ModelAdapterError, match="sum to 1\\.0"):
        adapter.predict_proba(["sample text"])


def test_sklearn_adapter_default_estimator_has_optional_extra_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "sklearn" or name.startswith("sklearn."):
            raise ModuleNotFoundError("No module named 'sklearn'", name="sklearn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("active_learning_sdk.adapters.sklearn", None)

    module = importlib.import_module("active_learning_sdk.adapters.sklearn")

    with pytest.raises(ImportError, match=r"active-learning-sdk\[sklearn\]"):
        module.SklearnTextClassifierAdapter()
