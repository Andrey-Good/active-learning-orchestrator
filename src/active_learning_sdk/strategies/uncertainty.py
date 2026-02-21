from __future__ import annotations

"""
Built-in sampling strategies (MVP).

These strategies implement common "uncertainty sampling" heuristics:
- entropy: pick items with highest prediction entropy
- margin: pick items where top-2 classes are close
- least_confidence: pick items with low max probability
- random: baseline
"""

import random
from typing import List, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine import SelectionContext


class RandomStrategy:
    """
    Pick k random ids from the pool.

    Attributes:
        name (str):
            Where: used by `StrategyScheduler._get_strategy()` and by user configs like `SchedulerConfig(strategy=...)`.
            What: stable string identifier for this strategy ("random").
            Why: lets the engine select strategies by name and makes configs serializable.
    """
    name = "random"

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> List[str]:
        if k <= 0:
            return []
        if len(pool_ids) <= k:
            return list(pool_ids)
        return random.sample(list(pool_ids), k)


class EntropyStrategy:
    """
    Pick items with the highest entropy of predicted probabilities.

    Higher entropy means the model is "more unsure".

    Attributes:
        name (str):
            Where: used by scheduler and configs to refer to this strategy.
            What: stable string identifier ("entropy").
            Why: ensures the strategy can be selected without importing the class directly.
    """
    name = "entropy"

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> List[str]:
        if k <= 0 or not pool_ids:
            return []
        import math

        probabilities = context.predict_proba(pool_ids)
        scored: List[Tuple[str, float]] = []
        for sample_id, probability in zip(pool_ids, probabilities):
            entropy = 0.0
            for value in list(probability):
                prob = float(value)
                if prob > 0:
                    entropy -= prob * math.log(prob + 1e-12)
            scored.append((sample_id, entropy))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [sample_id for sample_id, _ in scored[:k]]


class LeastConfidenceStrategy:
    """
    Pick items where the model is least confident (1 - max_proba).

    Attributes:
        name (str):
            Where: used by scheduler and configs to refer to this strategy.
            What: stable string identifier ("least_confidence").
            Why: makes it possible to choose this heuristic via config.
    """
    name = "least_confidence"

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> List[str]:
        if k <= 0 or not pool_ids:
            return []
        probabilities = context.predict_proba(pool_ids)
        scored: List[Tuple[str, float]] = []
        for sample_id, probability in zip(pool_ids, probabilities):
            probs = [float(value) for value in list(probability)]
            score = 1.0 - max(probs) if probs else 0.0
            scored.append((sample_id, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [sample_id for sample_id, _ in scored[:k]]


class MarginStrategy:
    """
    Pick items with the smallest margin between top-2 probabilities.

    Small margin means the model is torn between two classes.

    Attributes:
        name (str):
            Where: used by scheduler and configs to refer to this strategy.
            What: stable string identifier ("margin").
            Why: makes this heuristic selectable via `SchedulerConfig`.
    """
    name = "margin"

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> List[str]:
        if k <= 0 or not pool_ids:
            return []
        probabilities = context.predict_proba(pool_ids)
        scored: List[Tuple[str, float]] = []
        for sample_id, probability in zip(pool_ids, probabilities):
            probs = sorted([float(value) for value in list(probability)], reverse=True)
            if len(probs) >= 2:
                margin = probs[0] - probs[1]
            elif len(probs) == 1:
                margin = probs[0]
            else:
                margin = 1.0
            scored.append((sample_id, -margin))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [sample_id for sample_id, _ in scored[:k]]


class KCenterGreedyStrategy:
    """
    Placeholder for a diversity strategy (requires embeddings).

    This is intentionally not implemented in the scaffold.

    Attributes:
        name (str):
            Where: used by scheduler and configs to refer to this strategy.
            What: stable string identifier ("coreset_kcenter").
            Why: keeps the placeholder wired into the same selection mechanism as other strategies.
    """
    name = "coreset_kcenter"

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> List[str]:
        if k <= 0 or not pool_ids:
            return []
        raise NotImplementedError("KCenterGreedyStrategy is a scaffold; implement in active-learning-sdk[faiss].")
