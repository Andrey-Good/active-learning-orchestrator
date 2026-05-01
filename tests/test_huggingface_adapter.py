from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, Sequence

import pytest

from active_learning_sdk.adapters.base import inspect_model_capabilities
from active_learning_sdk.adapters.huggingface import HFSequenceClassifierAdapter
from active_learning_sdk.exceptions import ModelAdapterError


class _FakeNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class _FakeLogits:
    def __init__(self, row_count: int) -> None:
        self.row_count = row_count


class _FakeProbabilities:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def cpu(self) -> "_FakeProbabilities":
        return self

    def tolist(self) -> list[list[float]]:
        return self._rows


class _FakeFiniteResult:
    def __init__(self, value: bool) -> None:
        self.value = value

    def all(self) -> "_FakeFiniteResult":
        return self

    def item(self) -> bool:
        return self.value


class _FakeTokenizer:
    def __call__(self, texts: Sequence[str], **kwargs: Any) -> dict[str, Any]:
        return {"input_ids": list(texts)}


class _FakeHFModel:
    def eval(self) -> None:
        return None

    def __call__(self, **encoded: Any) -> SimpleNamespace:
        return SimpleNamespace(logits=_FakeLogits(len(encoded["input_ids"])))


@pytest.fixture(autouse=True)
def _fake_transformers_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace())


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch, rows: list[list[float]]) -> None:
    fake_torch = SimpleNamespace(
        no_grad=lambda: _FakeNoGrad(),
        softmax=lambda logits, dim: _FakeProbabilities(rows[: logits.row_count]),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_hf_predict_proba_rejects_non_finite_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = SimpleNamespace(
        isfinite=lambda logits: _FakeFiniteResult(False),
        no_grad=lambda: _FakeNoGrad(),
        softmax=lambda logits, dim: _FakeProbabilities([[0.5, 0.5]]),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    adapter = HFSequenceClassifierAdapter(model=_FakeHFModel(), tokenizer=_FakeTokenizer())

    with pytest.raises(ModelAdapterError, match="non-finite logits"):
        adapter.predict_proba(["sample"])


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([[float("nan"), 1.0]], "non-finite"),
        ([[-0.1, 1.1]], "negative"),
        ([[0.2, 0.2]], "sum to 1\\.0"),
        ([[1.0]], "at least two class columns"),
    ],
)
def test_hf_predict_proba_rejects_invalid_probability_rows(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[list[float]],
    message: str,
) -> None:
    _install_fake_torch(monkeypatch, rows)
    adapter = HFSequenceClassifierAdapter(model=_FakeHFModel(), tokenizer=_FakeTokenizer())

    with pytest.raises(ModelAdapterError, match=message):
        adapter.predict_proba(["sample"])


class _FakeTensor:
    def __init__(self, value: Any) -> None:
        self.value = value
        self.moved_to: Any = None

    def __len__(self) -> int:
        return len(self.value)

    def to(self, device: Any) -> "_FakeTensor":
        self.moved_to = device
        return self


class _DeviceTokenizer:
    def __init__(self) -> None:
        self.input_ids = _FakeTensor(["first", "second"])
        self.attention_mask = _FakeTensor([1, 1])

    def __call__(self, texts: Sequence[str], **kwargs: Any) -> dict[str, Any]:
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}


class _DeviceModel:
    device = "cuda:7"

    def __init__(self) -> None:
        self.seen_input_ids: _FakeTensor | None = None
        self.seen_attention_mask: _FakeTensor | None = None

    def eval(self) -> None:
        return None

    def __call__(self, **encoded: Any) -> SimpleNamespace:
        self.seen_input_ids = encoded["input_ids"]
        self.seen_attention_mask = encoded["attention_mask"]
        return SimpleNamespace(logits=_FakeLogits(len(encoded["input_ids"])))


def test_hf_predict_proba_moves_tokenizer_outputs_to_model_device(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch, [[0.25, 0.75], [0.6, 0.4]])
    model = _DeviceModel()
    tokenizer = _DeviceTokenizer()
    adapter = HFSequenceClassifierAdapter(model=model, tokenizer=tokenizer)

    assert adapter.predict_proba(["first", "second"]) == [[0.25, 0.75], [0.6, 0.4]]
    assert tokenizer.input_ids.moved_to == "cuda:7"
    assert tokenizer.attention_mask.moved_to == "cuda:7"
    assert model.seen_input_ids is tokenizer.input_ids
    assert model.seen_attention_mask is tokenizer.attention_mask


def test_hf_scaffold_reports_fit_and_evaluate_as_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    caps = inspect_model_capabilities(HFSequenceClassifierAdapter(model=_FakeHFModel(), tokenizer=_FakeTokenizer()))

    assert caps.predict_proba is True
    assert caps.fit is False
    assert caps.evaluate is False
