"""
Real transformer (DistilBERT) active-learning adapter.

Unlike :class:`HFSequenceClassifierAdapter` (a ``predict_proba``-only scaffold), this
adapter implements a full AL-ready contract on top of a Hugging Face
``AutoModelForSequenceClassification``:

- ``fit``             : cold-restart fine-tune per round (fair AL protocol), fp16 autocast on CUDA.
- ``evaluate``        : accuracy / macro-F1 / weighted-F1 / balanced-accuracy / macro-recall.
- ``predict_proba``   : softmax class probabilities, renormalized to sum exactly 1.0.
- ``embed``           : mean-pooled last hidden state (for coreset / diversity strategies).
- ``gradient_embed``  : closed-form BADGE gradient embeddings ``(p - onehot(y_hat)) ⊗ z``,
                        optionally random-projected for high class counts.

It is intentionally optimized for small AL pools on a single modest GPU (e.g. a 4 GB
laptop card) and for fast Kaggle T4 runs: fp16 autocast, batched inference under
``inference_mode``, and a per-round ``get_model_id`` so the engine's prediction/embedding
caches invalidate correctly after each ``fit``.

``predict_stochastic`` (MC-dropout) and ``predict_committee`` (seed ensemble) are deliberately
NOT implemented here yet; strategies that need them are out of scope for the deadline pilot and
the engine's capability detection will report them as unsupported.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from ..exceptions import ModelAdapterError
from .huggingface import HFSequenceClassifierAdapter, _ensure_huggingface_extra


# Flattened BADGE gradient embeddings ``num_labels * hidden_dim`` can become large for
# many-class datasets (e.g. Banking77: 77*768 ≈ 59k). Above this width we random-project
# to ``_BADGE_PROJECTION_DIM`` to keep greedy k-center fast and memory-bounded. Datasets with
# few classes (AG News 4, TREC 6) stay in their native gradient space.
_BADGE_PROJECTION_THRESHOLD = 4096
_BADGE_PROJECTION_DIM = 256


class DistilBERTALAdapter(HFSequenceClassifierAdapter):
    """Fine-tunable transformer adapter for active-learning benchmarks."""

    def __init__(
        self,
        labels: Sequence[str],
        *,
        model_name: str = "distilbert-base-uncased",
        seed: int = 13,
        lr: float = 2e-5,
        epochs_per_round: int = 3,
        train_batch_size: int = 16,
        eval_batch_size: int = 64,
        max_length: int = 128,
        warm_start: bool = False,
        shrink: float = 0.9,
        perturb: float = 0.01,
        embed_mode: str = "mean",
        device: str | None = None,
    ) -> None:
        _ensure_huggingface_extra()
        import torch  # type: ignore
        from transformers import AutoTokenizer  # type: ignore

        if len(labels) < 2:
            raise ModelAdapterError("DistilBERTALAdapter requires at least two labels.")
        if embed_mode not in {"mean", "cls"}:
            raise ModelAdapterError("embed_mode must be 'mean' or 'cls'.")

        self.labels: List[str] = [str(label) for label in labels]
        self._label_to_id: Dict[str, int] = {label: index for index, label in enumerate(self.labels)}
        self.num_labels = len(self.labels)

        self.model_name = model_name
        self.seed = int(seed)
        self.lr = float(lr)
        self.epochs_per_round = int(epochs_per_round)
        self.train_batch_size = int(train_batch_size)
        self.eval_batch_size = int(eval_batch_size)
        self.max_length = int(max_length)
        self.warm_start = bool(warm_start)
        self.shrink = float(shrink)
        self.perturb = float(perturb)
        self.embed_mode = embed_mode

        if device is not None:
            self._device = torch.device(device)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = None  # built fresh on each fit() (cold restart) or reused (warm start)
        self._round = 0
        self._fitted = False

    # ------------------------------------------------------------------ helpers

    def _build_model(self) -> Any:
        import torch  # type: ignore
        from transformers import AutoModelForSequenceClassification  # type: ignore

        torch.manual_seed(self.seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        return model.to(self._device)

    def _encode(self, texts: Sequence[str]) -> Any:
        return self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def _autocast(self):
        import torch  # type: ignore
        from contextlib import nullcontext

        if self._device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _backbone(self) -> Any:
        """Return the encoder sub-module (without the classification head)."""
        # DistilBERT exposes ``.distilbert``; BERT/RoBERTa expose ``.bert``/``.roberta``.
        for attr in ("distilbert", "bert", "roberta", "base_model"):
            module = getattr(self.model, attr, None)
            if module is not None:
                return module
        return self.model

    # ------------------------------------------------------------------ fit / evaluate

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs: Any) -> None:
        import torch  # type: ignore
        from torch.optim import AdamW  # type: ignore

        text_list = list(texts)
        if not text_list:
            raise ModelAdapterError("DistilBERTALAdapter.fit received no training texts.")
        try:
            label_ids = [self._label_to_id[str(label)] for label in labels]
        except KeyError as error:
            raise ModelAdapterError(f"DistilBERTALAdapter.fit received unknown label {error}.") from error

        if self.model is None or not self.warm_start:
            self.model = self._build_model()  # cold restart: fresh weights every round
        else:
            # Warm path: model already exists.  Apply shrink-and-perturb so the optimizer
            # escapes local minima without fully discarding the prior fine-tune.
            # w <- shrink*w + N(0, perturb * std(w)), seeded deterministically per round.
            import torch  # type: ignore

            gen = torch.Generator().manual_seed(self.seed + self._round)
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.requires_grad:
                        std = float(param.std()) if param.numel() > 1 else 0.0
                        noise = torch.zeros_like(param).normal_(
                            mean=0.0, std=max(self.perturb * std, 1e-12), generator=gen
                        )
                        param.mul_(self.shrink).add_(noise)

        self.model.train()
        encoded = self._encode(text_list)
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)
        target = torch.tensor(label_ids, dtype=torch.long, device=self._device)

        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        use_cuda = self._device.type == "cuda"
        try:  # torch>=2.4 moved GradScaler to torch.amp; fall back for older versions
            scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
        n = len(text_list)
        generator = torch.Generator().manual_seed(self.seed + self._round)

        for _ in range(self.epochs_per_round):
            order = torch.randperm(n, generator=generator)
            for start in range(0, n, self.train_batch_size):
                idx = order[start : start + self.train_batch_size]
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    output = self.model(
                        input_ids=input_ids[idx],
                        attention_mask=attention_mask[idx],
                        labels=target[idx],
                    )
                    loss = output.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        self.model.eval()
        self._fitted = True
        self._round += 1

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> Dict[str, float]:
        from sklearn.metrics import (  # type: ignore
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            recall_score,
        )

        true_labels = [str(label) for label in labels]
        proba = self.predict_proba(texts, batch_size=self.eval_batch_size)
        predictions = [self.labels[max(range(len(row)), key=row.__getitem__)] for row in proba]
        return {
            "accuracy": float(accuracy_score(true_labels, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(true_labels, predictions)),
            "macro_f1": float(f1_score(true_labels, predictions, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(true_labels, predictions, average="weighted", zero_division=0)),
            "macro_recall": float(recall_score(true_labels, predictions, average="macro", zero_division=0)),
        }

    # ------------------------------------------------------------------ inference

    def _ensure_fitted(self) -> None:
        if self.model is None:
            raise ModelAdapterError("DistilBERTALAdapter must be fit() before inference.")

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> List[List[float]]:
        import torch  # type: ignore

        self._ensure_fitted()
        text_list = list(texts)
        size = max(1, int(batch_size))
        self.model.eval()
        rows: List[List[float]] = []
        with torch.inference_mode():
            for offset in range(0, len(text_list), size):
                chunk = text_list[offset : offset + size]
                encoded = self._encode(chunk)
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                with self._autocast():
                    logits = self.model(**encoded).logits
                probabilities = torch.softmax(logits.float(), dim=-1).cpu().tolist()
                rows.extend(self._renormalize(row) for row in probabilities)
        return rows

    @staticmethod
    def _renormalize(row: Sequence[float]) -> List[float]:
        # Guarantee an exact-sum-to-1.0 row in float64 so downstream validators never reject
        # fp16-accumulated softmax output.
        values = [float(max(0.0, value)) for value in row]
        total = sum(values)
        if total <= 0.0:
            uniform = 1.0 / len(values)
            return [uniform for _ in values]
        return [value / total for value in values]

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> List[List[float]]:
        import torch  # type: ignore

        self._ensure_fitted()
        text_list = list(texts)
        size = max(1, int(batch_size))
        backbone = self._backbone()
        self.model.eval()
        out: List[List[float]] = []
        with torch.inference_mode():
            for offset in range(0, len(text_list), size):
                chunk = text_list[offset : offset + size]
                encoded = self._encode(chunk)
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                with self._autocast():
                    hidden = backbone(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                    ).last_hidden_state
                pooled = self._pool(hidden.float(), encoded["attention_mask"])
                out.extend(pooled.cpu().tolist())
        return out

    def _pool(self, hidden: Any, attention_mask: Any) -> Any:
        if self.embed_mode == "cls":
            return hidden[:, 0, :]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def gradient_embed(
        self,
        texts: Sequence[str],
        labels: Sequence[Any] | None = None,
        batch_size: int = 32,
    ) -> List[List[float]]:
        """
        Closed-form BADGE gradient embedding of the final linear layer.

        For each sample the loss gradient w.r.t. the classifier weights factorizes as
        ``g = (p - onehot(y_hat)) ⊗ z`` where ``p`` is the predicted distribution, ``y_hat``
        the (pseudo-)label, and ``z`` the pooled penultimate representation. We compute this
        directly (no autograd) and flatten to ``num_labels * hidden_dim``; for many-class
        datasets we random-project to a fixed lower dimension.
        """
        import torch  # type: ignore

        self._ensure_fitted()
        text_list = list(texts)
        size = max(1, int(batch_size))
        backbone = self._backbone()
        self.model.eval()

        provided_ids: List[int] | None = None
        if labels is not None:
            provided_ids = [self._label_to_id[str(label)] for label in labels]

        embeddings: List[Any] = []
        projection: Any | None = None
        cursor = 0
        with torch.inference_mode():
            for offset in range(0, len(text_list), size):
                chunk = text_list[offset : offset + size]
                encoded = self._encode(chunk)
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                with self._autocast():
                    hidden = backbone(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                    ).last_hidden_state
                    logits = self.model(**encoded).logits
                pooled = self._pool(hidden.float(), encoded["attention_mask"])  # [B, H]
                probs = torch.softmax(logits.float(), dim=-1)  # [B, C]
                if provided_ids is None:
                    target_ids = probs.argmax(dim=-1)
                else:
                    target_ids = torch.tensor(
                        provided_ids[cursor : cursor + len(chunk)],
                        dtype=torch.long,
                        device=probs.device,
                    )
                residual = probs.clone()
                residual[torch.arange(residual.size(0)), target_ids] -= 1.0  # [B, C]
                # Outer product (p - onehot) ⊗ z  ->  [B, C, H]  ->  [B, C*H]
                grad = torch.bmm(residual.unsqueeze(2), pooled.unsqueeze(1))
                grad = grad.reshape(grad.size(0), -1)
                if grad.size(1) > _BADGE_PROJECTION_THRESHOLD:
                    if projection is None:
                        generator = torch.Generator(device=grad.device).manual_seed(self.seed)
                        projection = torch.randn(
                            grad.size(1), _BADGE_PROJECTION_DIM,
                            generator=generator, device=grad.device,
                        ) / (_BADGE_PROJECTION_DIM ** 0.5)
                    grad = grad @ projection
                embeddings.extend(grad.cpu().tolist())
                cursor += len(chunk)
        return embeddings

    # ------------------------------------------------------------------ identity

    def get_model_id(self) -> str:
        return f"distilbert-{self.model_name}-r{self._round}-s{self.seed}"

    def get_embedding_config(self) -> str:
        return f"embed-{self.embed_mode}-r{self._round}"
