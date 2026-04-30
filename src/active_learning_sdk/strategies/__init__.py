"""Sampling strategy contracts and built-ins."""

from .base import SamplingStrategy
from .adaptive import AdaptiveUncertaintyDiversityStrategy
from .badge import BadgeStrategy
from .embedding import (
    DeduplicateNearNeighborsStrategy,
    DensityWeightedDiversityStrategy,
    EmbeddingKMeansPPStrategy,
    KCenterGreedyStrategy,
    MaxMinEmbeddingStrategy,
)
from .hybrid import HybridStrategy, normalize_scores, validate_hybrid_config
from .stochastic import (
    BaldStrategy,
    CommitteeKLDivergenceStrategy,
    CommitteeMarginStrategy,
    CommitteePairwiseDisagreementStrategy,
    CommitteeVoteEntropyStrategy,
    McDropoutEntropyStrategy,
    PredictionVarianceStrategy,
    VariationRatioStrategy,
)
from .uncertainty import (
    ClassBalancedEntropyStrategy,
    ClassGroupBalancedEntropyStrategy,
    EntropyStrategy,
    GroupDiverseEntropyStrategy,
    LeastConfidenceStrategy,
    MarginStrategy,
    RandomStrategy,
)

__all__ = [
    "SamplingStrategy",
    "AdaptiveUncertaintyDiversityStrategy",
    "BadgeStrategy",
    "RandomStrategy",
    "ClassBalancedEntropyStrategy",
    "ClassGroupBalancedEntropyStrategy",
    "EntropyStrategy",
    "GroupDiverseEntropyStrategy",
    "LeastConfidenceStrategy",
    "MarginStrategy",
    "KCenterGreedyStrategy",
    "EmbeddingKMeansPPStrategy",
    "MaxMinEmbeddingStrategy",
    "DeduplicateNearNeighborsStrategy",
    "DensityWeightedDiversityStrategy",
    "HybridStrategy",
    "normalize_scores",
    "validate_hybrid_config",
    "McDropoutEntropyStrategy",
    "BaldStrategy",
    "VariationRatioStrategy",
    "PredictionVarianceStrategy",
    "CommitteeVoteEntropyStrategy",
    "CommitteeKLDivergenceStrategy",
    "CommitteePairwiseDisagreementStrategy",
    "CommitteeMarginStrategy",
]
