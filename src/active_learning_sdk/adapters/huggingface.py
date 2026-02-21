from __future__ import annotations

from typing import Any, Dict, List, Sequence

from ..exceptions import ModelAdapterError


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

        self.model.eval()
        output: List[List[float]] = []
        with torch.no_grad():
            for offset in range(0, len(texts), max(1, batch_size)):
                chunk = list(texts[offset : offset + batch_size])
                encoded = self.tokenizer(chunk, padding=True, truncation=True, return_tensors="pt")
                logits = self.model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
                output.extend([[float(value) for value in row] for row in probabilities])
        return output

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs) -> None:
        """Train/update the model. This is intentionally not implemented in the scaffold."""
        raise NotImplementedError("Provide your own training loop and override fit().")

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
