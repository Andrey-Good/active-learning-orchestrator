"""Adaptive acquisition strategies for practical default active-learning loops."""

from __future__ import annotations

from typing import List, Sequence, TYPE_CHECKING

from ._shared import label_count as _shared_label_count
from .hybrid import HybridStrategy
from .uncertainty import EntropyStrategy

if TYPE_CHECKING:
    from ..engine import SelectionContext

_MANY_CLASS_MIN_LABELS = 20
_DIVERSITY_PREFILTER_UNCERTAINTY_CONFIG = {
    "mode": "diversity_prefilter_uncertainty",
    "uncertainty": "entropy",
    "diversity": "coreset_kcenter",
    "prefilter_multiplier": 3.0,
}
_DIVERSITY_PREFILTER_EFFECTIVE_STRATEGY = "hybrid:diversity_prefilter_uncertainty:coreset_kcenter+entropy"


class AdaptiveUncertaintyDiversityStrategy:
    """Adapt uncertainty/diversity defaults to label-space shape.

    Many-class text workflows use diversity-prefiltered uncertainty across
    acquisition rounds when embeddings are available. Smaller label spaces start
    with guarded uncertainty/diversity, then switch to entropy. These choices are
    based only on already-labeled sample count, public capabilities, and the
    public label schema, so the strategy does not inspect unlabeled oracle
    labels.
    """

    name = "adaptive_uncertainty_diversity"
    required_capabilities = frozenset({"predict_proba", "embed"})

    def __init__(self, *, early_label_multiplier: int = 8) -> None:
        self.early_label_multiplier = max(1, int(early_label_multiplier))

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> List[str]:
        if k <= 0 or not pool_ids:
            return []
        if self._use_many_class_diversity_prefilter(context):
            result = self._diversity_prefilter_uncertainty().select(pool_ids, k, context)
            self._record_strategy_diagnostic(
                context,
                phase="many_class_diversity_prefilter",
                effective_strategy=_DIVERSITY_PREFILTER_EFFECTIVE_STRATEGY,
            )
            return result.selected
        if self._use_early_guarded_phase(context):
            result = HybridStrategy(
                {
                    "mode": "weighted",
                    "uncertainty": "entropy",
                    "diversity": "coreset_kcenter",
                    "uncertainty_weight": 0.5,
                    "diversity_weight": 0.5,
                    "class_balance": True,
                    "group_balance": True,
                    "exploration_fraction": 0.2,
                }
            ).select(pool_ids, k, context)
            self._record_strategy_diagnostic(
                context,
                phase="early_guarded_hybrid",
                effective_strategy="hybrid:weighted:coreset_kcenter+entropy",
            )
            return result.selected
        selected = EntropyStrategy().select(pool_ids, k, context)
        self._record_strategy_diagnostic(
            context,
            phase="mature_entropy",
            effective_strategy=EntropyStrategy.name,
        )
        return selected

    def _label_count(self, context: "SelectionContext") -> int:
        return _shared_label_count(context) or 0

    def _labeled_count(self, context: "SelectionContext") -> int:
        return len(getattr(context, "labeled_ids", []))

    def _diversity_prefilter_uncertainty(self) -> HybridStrategy:
        return HybridStrategy(_DIVERSITY_PREFILTER_UNCERTAINTY_CONFIG)

    def _use_many_class_diversity_prefilter(self, context: "SelectionContext") -> bool:
        label_count = self._label_count(context)
        if label_count < _MANY_CLASS_MIN_LABELS:
            return False
        return callable(getattr(context, "embed", None))

    def _use_early_guarded_phase(self, context: "SelectionContext") -> bool:
        labeled_count = self._labeled_count(context)
        label_count = self._label_count(context)
        switch_after = max(32, label_count * self.early_label_multiplier)
        return labeled_count < switch_after

    def _record_strategy_diagnostic(
        self,
        context: "SelectionContext",
        *,
        phase: str,
        effective_strategy: str,
    ) -> None:
        recorder = getattr(context, "record_strategy_diagnostic", None)
        if not callable(recorder):
            return
        label_count = self._label_count(context)
        recorder(
            self.name,
            {
                "phase": phase,
                "label_count": label_count,
                "labeled_count": self._labeled_count(context),
                "effective_strategy": effective_strategy,
            },
        )
