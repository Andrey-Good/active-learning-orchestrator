"""Adaptive acquisition strategies for practical default active-learning loops."""

from __future__ import annotations


from typing import List, Sequence, TYPE_CHECKING

from .hybrid import HybridStrategy
from .uncertainty import EntropyStrategy, RandomStrategy

if TYPE_CHECKING:
    from ..engine import SelectionContext

_MANY_CLASS_MIN_LABELS = 20
_MANY_CLASS_EXPLORATION_MULTIPLIER = 1.25


class AdaptiveUncertaintyDiversityStrategy:
    """Start with guarded uncertainty/diversity, then switch to entropy.

    Early active-learning rounds benefit from exploration and group/class
    guardrails, while later rounds can exploit model uncertainty more directly.
    The switch is based only on already-labeled sample count and the public label
    schema, so the strategy does not inspect unlabeled oracle labels.
    """

    name = "adaptive_uncertainty_diversity"
    required_capabilities = frozenset({"predict_proba", "embed"})

    def __init__(self, *, early_label_multiplier: int = 8) -> None:
        self.early_label_multiplier = max(1, int(early_label_multiplier))

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> List[str]:
        if k <= 0 or not pool_ids:
            return []
        if self._use_many_class_random_exploration(context):
            return RandomStrategy().select(pool_ids, k, context)
        if self._use_many_class_diversity_prefilter(context):
            return HybridStrategy(
                {
                    "mode": "diversity_prefilter_uncertainty",
                    "uncertainty": "entropy",
                    "diversity": "coreset_kcenter",
                    "prefilter_multiplier": 3.0,
                }
            ).select(pool_ids, k, context).selected
        if self._use_early_guarded_phase(context):
            return HybridStrategy(
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
            ).select(pool_ids, k, context).selected
        return EntropyStrategy().select(pool_ids, k, context)

    def _label_count(self, context: "SelectionContext") -> int:
        label_schema = getattr(context, "label_schema", None)
        labels = getattr(label_schema, "labels", None)
        try:
            return len(list(labels)) if labels is not None else 0
        except TypeError:
            return 0

    def _labeled_count(self, context: "SelectionContext") -> int:
        return len(getattr(context, "labeled_ids", []))

    def _use_many_class_random_exploration(self, context: "SelectionContext") -> bool:
        label_count = self._label_count(context)
        if label_count < _MANY_CLASS_MIN_LABELS:
            return False
        exploration_until = max(32, int(label_count * _MANY_CLASS_EXPLORATION_MULTIPLIER))
        return self._labeled_count(context) < exploration_until

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
