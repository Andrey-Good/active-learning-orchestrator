from __future__ import annotations

import math
import sys
from importlib.util import find_spec
from numbers import Real
from typing import Any, Dict, List, Sequence

from ..exceptions import ModelAdapterError
from .base import normalize_adapter_batch_size, unsupported_adapter_method


def _huggingface_import_error() -> ImportError:
    return ImportError(
        "HFSequenceClassifierAdapter requires optional Hugging Face dependencies. "
        "Install them with active-learning-sdk[huggingface]."
    )


def _ensure_huggingface_extra() -> None:
    missing = [name for name in ("torch", "transformers") if not _dependency_available(name)]
    if missing:
        raise _huggingface_import_error()


def _dependency_available(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return find_spec(name) is not None
    except ValueError:
        return True


class HFSequenceClassifierAdapter:
    """
    Lightweight HuggingFace sequence-classification adapter scaffold.

    This wrapper exposes the SDK adapter contract. Training/evaluation hooks are kept
    minimal in the scaffold and should be specialized per training stack.

    Attributes:
        model (Any):
            Where: called in `predict_proba()` to produce logits.
            What: a HuggingFace `AutoModelForSequenceClassification`-like object.
            Why: the SDK talks to models through adapters; this stores the real HF model.
        tokenizer (Any):
            Where: used in `predict_proba()` to tokenize input texts.
            What: a HuggingFace tokenizer compatible with the model.
            Why: tokenization is required before the model forward pass.
    """

    def __init__(self, model: Any, tokenizer: Any) -> None:
        _ensure_huggingface_extra()
        self.model = model
        self.tokenizer = tokenizer

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> List[List[float]]:
        """
        Predict class probabilities for a list of texts.

        This method is implemented because it is useful for uncertainty strategies.
        Training (`fit`) and evaluation (`evaluate`) are left as user code in the scaffold.
        """
        try:
            import torch  # type: ignore
        except Exception as error:
            raise ModelAdapterError("HFSequenceClassifierAdapter requires torch.") from error

        text_list = list(texts)
        size = self._normalized_batch_size(batch_size)
        self.model.eval()
        output: List[List[float]] = []
        with torch.no_grad():
            for offset in range(0, len(text_list), size):
                chunk = text_list[offset : offset + size]
                encoded = self.tokenizer(chunk, padding=True, truncation=True, return_tensors="pt")
                encoded = self._move_encoded_to_model_device(encoded)
                logits = self.model(**encoded).logits
                self._validate_logits(logits, torch)
                probabilities = torch.softmax(logits, dim=-1)
                probability_rows = self._tensor_rows(probabilities)
                output.extend(self._validate_probability_rows(probability_rows, expected_row_count=len(chunk)))
        if len(output) != len(text_list):
            raise ModelAdapterError(
                f"HFSequenceClassifierAdapter returned {len(output)} probability rows "
                f"for {len(text_list)} input texts."
            )
        return output

    def _move_encoded_to_model_device(self, encoded: Any) -> Any:
        device = self._model_device()
        if device is None:
            return encoded

        encoded_to = getattr(encoded, "to", None)
        if callable(encoded_to):
            return encoded_to(device)

        if isinstance(encoded, dict):
            moved: Dict[str, Any] = {}
            for key, value in encoded.items():
                value_to = getattr(value, "to", None)
                moved[key] = value_to(device) if callable(value_to) else value
            return moved
        return encoded

    def _model_device(self) -> Any:
        device = getattr(self.model, "device", None)
        if device is not None:
            return device

        parameters = getattr(self.model, "parameters", None)
        if not callable(parameters):
            return None
        try:
            first_parameter = next(iter(parameters()))
        except StopIteration:
            return None
        except Exception:
            return None
        return getattr(first_parameter, "device", None)

    @staticmethod
    def _validate_logits(logits: Any, torch: Any) -> None:
        isfinite = getattr(torch, "isfinite", None)
        if not callable(isfinite):
            return
        try:
            finite_result = isfinite(logits)
            all_result = finite_result.all()
            item = getattr(all_result, "item", None)
            all_finite = bool(item() if callable(item) else all_result)
        except Exception:
            return
        if not all_finite:
            raise ModelAdapterError("HFSequenceClassifierAdapter returned non-finite logits.")

    @staticmethod
    def _tensor_rows(probabilities: Any) -> List[Any]:
        detach = getattr(probabilities, "detach", None)
        if callable(detach):
            probabilities = detach()
        cpu = getattr(probabilities, "cpu", None)
        if callable(cpu):
            probabilities = cpu()
        tolist = getattr(probabilities, "tolist", None)
        if not callable(tolist):
            raise ModelAdapterError("HFSequenceClassifierAdapter probabilities must provide tolist().")
        rows = tolist()
        if isinstance(rows, (str, bytes)):
            raise ModelAdapterError("HFSequenceClassifierAdapter probabilities must be a 2D matrix.")
        try:
            return list(rows)
        except TypeError as error:
            raise ModelAdapterError("HFSequenceClassifierAdapter probabilities must be a 2D matrix.") from error

    @staticmethod
    def _validate_probability_rows(probabilities: Sequence[Any], *, expected_row_count: int) -> List[List[float]]:
        rows = list(probabilities)
        if len(rows) != expected_row_count:
            raise ModelAdapterError(
                f"HFSequenceClassifierAdapter returned {len(rows)} probability rows "
                f"for {expected_row_count} input texts in a batch."
            )

        validated: List[List[float]] = []
        for row_index, row in enumerate(rows):
            if isinstance(row, (str, bytes)):
                raise ModelAdapterError(
                    f"HFSequenceClassifierAdapter probability row {row_index} must be a sequence."
                )
            try:
                values = list(row)
            except TypeError as error:
                raise ModelAdapterError(
                    f"HFSequenceClassifierAdapter probability row {row_index} must be a sequence."
                ) from error
            if len(values) < 2:
                raise ModelAdapterError(
                    "HFSequenceClassifierAdapter probability rows must contain at least two class columns."
                )

            cleaned: List[float] = []
            for column_index, value in enumerate(values):
                if isinstance(value, bool) or not isinstance(value, (int, float, Real)):
                    raise ModelAdapterError(
                        f"HFSequenceClassifierAdapter probability row {row_index}, column {column_index} "
                        "must be numeric."
                    )
                probability = float(value)
                if not math.isfinite(probability):
                    raise ModelAdapterError("HFSequenceClassifierAdapter returned non-finite probabilities.")
                if probability < 0:
                    raise ModelAdapterError("HFSequenceClassifierAdapter returned negative probabilities.")
                cleaned.append(probability)

            row_sum = sum(cleaned)
            if row_sum <= 0:
                raise ModelAdapterError("HFSequenceClassifierAdapter returned a row with non-positive probability sum.")
            if not math.isclose(row_sum, 1.0, rel_tol=1e-9, abs_tol=1e-12):
                raise ModelAdapterError(
                    f"HFSequenceClassifierAdapter probability rows must sum to 1.0; got {row_sum}."
                )
            validated.append(cleaned)
        return validated

    @staticmethod
    def _normalized_batch_size(batch_size: int) -> int:
        return normalize_adapter_batch_size(batch_size)

    @unsupported_adapter_method("placeholder scaffold method; override fit() with a real training loop")
    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs) -> None:
        """Train/update the model. This is intentionally not implemented in the scaffold."""
        raise NotImplementedError("Provide your own training loop and override fit().")

    @unsupported_adapter_method("placeholder scaffold method; override evaluate() with a real evaluation routine")
    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> Dict[str, float]:
        """Evaluate the model. This is intentionally not implemented in the scaffold."""
        raise NotImplementedError("Provide your own evaluation routine and override evaluate().")

    def get_model_id(self) -> str:
        """
        Return a stable identifier for caching.

        If your model changes over time (training), consider including a version in the id.
        """
        model_name = getattr(getattr(self.model, "config", None), "_name_or_path", None)
        if model_name:
            return str(model_name)
        return self.model.__class__.__name__
