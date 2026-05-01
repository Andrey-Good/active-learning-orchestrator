"""Stochastic prediction and committee disagreement strategies."""

from __future__ import annotations


import math
from collections import Counter
from numbers import Real
from typing import Any, Callable, List, Sequence, TYPE_CHECKING

from ..exceptions import ConfigurationError
from .uncertainty import _select_top_scored

if TYPE_CHECKING:
    from ..engine import SelectionContext


_PROBABILITY_SUM_REL_TOL = 1e-9
_PROBABILITY_SUM_ABS_TOL = 1e-12


def _unique_pool_ids(pool_ids: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for raw_sample_id in pool_ids:
        sample_id = str(raw_sample_id)
        if sample_id in seen:
            continue
        seen.add(sample_id)
        out.append(sample_id)
    return out


def _target_count(sample_ids: Sequence[str], k: int) -> int:
    if k <= 0 or not sample_ids:
        return 0
    return min(k, len(sample_ids))


def _entropy(probabilities: Sequence[float]) -> float:
    value = 0.0
    for probability in probabilities:
        if probability > 0.0:
            value -= probability * math.log(probability)
    return value


def _mean_probability(probabilities: Sequence[Sequence[float]]) -> List[float]:
    width = len(probabilities[0])
    return [sum(row[column] for row in probabilities) / len(probabilities) for column in range(width)]


def _argmax(probabilities: Sequence[float]) -> int:
    return max(range(len(probabilities)), key=lambda index: (probabilities[index], -index))


def _normalize_probability_cube(
    probabilities: Any,
    sample_ids: Sequence[str],
    *,
    strategy_name: str,
    method_name: str,
    expected_member_count: int | None = None,
    min_member_count: int | None = None,
    expected_width: int | None = None,
) -> List[List[List[float]]]:
    try:
        sample_rows = list(probabilities)
    except TypeError as exc:
        raise ConfigurationError(f"{strategy_name}.{method_name} output must be sample-row-like.") from exc

    if len(sample_rows) != len(sample_ids):
        raise ConfigurationError(
            f"{strategy_name}.{method_name} returned {len(sample_rows)} rows for {len(sample_ids)} sample ids."
        )

    normalized_samples: List[List[List[float]]] = []
    observed_width: int | None = None
    observed_member_count: int | None = None
    for sample_index, (sample_id, sample_row) in enumerate(zip(sample_ids, sample_rows)):
        if isinstance(sample_row, (str, bytes)):
            raise ConfigurationError(
                f"{strategy_name}.{method_name} sample row {sample_index} for sample {sample_id!r} "
                "must be a sequence of probability rows."
            )
        try:
            member_rows = list(sample_row)
        except TypeError as exc:
            raise ConfigurationError(
                f"{strategy_name}.{method_name} sample row {sample_index} for sample {sample_id!r} must be a sequence."
            ) from exc

        if not member_rows:
            raise ConfigurationError(
                f"{strategy_name}.{method_name} sample row {sample_index} for sample {sample_id!r} "
                "must contain at least one pass/member."
            )
        if expected_member_count is not None and len(member_rows) != expected_member_count:
            raise ConfigurationError(
                f"{strategy_name}.{method_name} sample row {sample_index} for sample {sample_id!r} "
                f"has {len(member_rows)} passes/members; expected {expected_member_count}."
            )
        if min_member_count is not None and len(member_rows) < min_member_count:
            raise ConfigurationError(
                f"{strategy_name}.{method_name} sample row {sample_index} for sample {sample_id!r} "
                f"has {len(member_rows)} passes/members; expected at least {min_member_count}."
            )
        if observed_member_count is None:
            observed_member_count = len(member_rows)
        elif len(member_rows) != observed_member_count:
            raise ConfigurationError(
                f"{strategy_name}.{method_name} sample row {sample_index} for sample {sample_id!r} "
                f"has {len(member_rows)} passes/members; expected {observed_member_count}."
            )

        normalized_members: List[List[float]] = []
        for member_index, member_row in enumerate(member_rows):
            if isinstance(member_row, (str, bytes)):
                raise ConfigurationError(
                    f"{strategy_name}.{method_name} probability row {sample_index}:{member_index} "
                    f"for sample {sample_id!r} must be a sequence of numeric probabilities."
                )
            try:
                values = list(member_row)
            except TypeError as exc:
                raise ConfigurationError(
                    f"{strategy_name}.{method_name} probability row {sample_index}:{member_index} "
                    f"for sample {sample_id!r} must be a sequence."
                ) from exc

            if not values:
                raise ConfigurationError(
                    f"{strategy_name}.{method_name} probability row {sample_index}:{member_index} "
                    f"for sample {sample_id!r} must not be empty."
                )
            if len(values) < 2:
                raise ConfigurationError(
                    f"{strategy_name}.{method_name} probability row {sample_index}:{member_index} "
                    f"for sample {sample_id!r} must have at least 2 probability columns."
                )

            if expected_width is not None and len(values) != expected_width:
                raise ConfigurationError(
                    f"{strategy_name}.{method_name} probability row {sample_index}:{member_index} "
                    f"for sample {sample_id!r} has width {len(values)}; expected label_schema width {expected_width}."
                )
            if observed_width is None:
                observed_width = len(values)
            elif len(values) != observed_width:
                raise ConfigurationError(
                    f"{strategy_name}.{method_name} probability row {sample_index}:{member_index} "
                    f"for sample {sample_id!r} has width {len(values)}; expected {observed_width}."
                )

            cleaned: List[float] = []
            for column_index, value in enumerate(values):
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ConfigurationError(
                        f"{strategy_name}.{method_name} value at row {sample_index}:{member_index}, "
                        f"column {column_index} must be numeric."
                    )
                probability = float(value)
                if not math.isfinite(probability):
                    raise ConfigurationError(
                        f"{strategy_name}.{method_name} value at row {sample_index}:{member_index}, "
                        f"column {column_index} must be finite."
                    )
                if probability < 0.0:
                    raise ConfigurationError(
                        f"{strategy_name}.{method_name} value at row {sample_index}:{member_index}, "
                        f"column {column_index} must be non-negative."
                    )
                cleaned.append(probability)

            row_sum = sum(cleaned)
            if row_sum <= 0.0:
                raise ConfigurationError(
                    f"{strategy_name}.{method_name} probability row {sample_index}:{member_index} "
                    f"for sample {sample_id!r} must have a positive sum."
                )
            if not math.isclose(row_sum, 1.0, rel_tol=_PROBABILITY_SUM_REL_TOL, abs_tol=_PROBABILITY_SUM_ABS_TOL):
                raise ConfigurationError(
                    f"{strategy_name}.{method_name} probability row {sample_index}:{member_index} "
                    f"for sample {sample_id!r} sums to {row_sum}; expected 1.0."
                )
            normalized_members.append(cleaned)

        normalized_samples.append(normalized_members)

    return normalized_samples


def _select_by_cube_scores(
    strategy_name: str,
    pool_ids: Sequence[str],
    k: int,
    context: "SelectionContext",
    cube: Sequence[Sequence[Sequence[float]]],
    scorer: Callable[[Sequence[Sequence[float]]], float],
) -> List[str]:
    sample_ids = _unique_pool_ids(pool_ids)
    target_k = _target_count(sample_ids, k)
    if target_k <= 0:
        return []
    scored = [(sample_id, scorer(rows)) for sample_id, rows in zip(sample_ids, cube)]
    return _select_top_scored(scored, target_k, strategy_name=strategy_name, context=context)


def _label_schema_width(context: "SelectionContext") -> int | None:
    label_schema = getattr(context, "label_schema", None)
    labels = getattr(label_schema, "labels", None)
    if labels is None:
        return None
    try:
        width = len(labels)
    except TypeError:
        return None
    return width if width > 0 else None


class _StochasticStrategyBase:
    name: str
    required_capabilities: frozenset[str] = frozenset({"predict_stochastic"})
    n_passes = 10
    batch_size = 32

    def score(self, probabilities: Sequence[Sequence[float]]) -> float:
        raise NotImplementedError

    def _probabilities(self, sample_ids: Sequence[str], context: "SelectionContext") -> List[List[List[float]]]:
        return _normalize_probability_cube(
            context.predict_stochastic(sample_ids, n=self.n_passes, batch_size=self.batch_size),
            sample_ids,
            strategy_name=self.name,
            method_name="predict_stochastic",
            expected_member_count=self.n_passes,
            expected_width=_label_schema_width(context),
        )

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> List[str]:
        sample_ids = _unique_pool_ids(pool_ids)
        target_k = _target_count(sample_ids, k)
        if target_k <= 0:
            return []
        cube = self._probabilities(sample_ids, context)
        return _select_by_cube_scores(self.name, sample_ids, target_k, context, cube, self.score)


class McDropoutEntropyStrategy(_StochasticStrategyBase):
    """Select samples with highest entropy of mean stochastic probability."""

    name = "mc_dropout_entropy"

    def score(self, probabilities: Sequence[Sequence[float]]) -> float:
        return _entropy(_mean_probability(probabilities))


class BaldStrategy(_StochasticStrategyBase):
    """Bayesian Active Learning by Disagreement over stochastic predictions."""

    name = "bald"

    def score(self, probabilities: Sequence[Sequence[float]]) -> float:
        mean_entropy = _entropy(_mean_probability(probabilities))
        member_entropy = sum(_entropy(row) for row in probabilities) / len(probabilities)
        return mean_entropy - member_entropy


class VariationRatioStrategy(_StochasticStrategyBase):
    """Select samples with the largest stochastic argmax vote variation."""

    name = "variation_ratio"

    def score(self, probabilities: Sequence[Sequence[float]]) -> float:
        votes = [_argmax(row) for row in probabilities]
        most_common_count = Counter(votes).most_common(1)[0][1]
        return 1.0 - (most_common_count / len(votes))


class PredictionVarianceStrategy(_StochasticStrategyBase):
    """Select samples with largest mean class-probability variance across passes."""

    name = "prediction_variance"

    def score(self, probabilities: Sequence[Sequence[float]]) -> float:
        means = _mean_probability(probabilities)
        width = len(means)
        variances = []
        for column in range(width):
            variances.append(sum((row[column] - means[column]) ** 2 for row in probabilities) / len(probabilities))
        return sum(variances) / width


class _CommitteeStrategyBase:
    name: str
    required_capabilities: frozenset[str] = frozenset({"predict_committee"})
    batch_size = 32

    def score(self, probabilities: Sequence[Sequence[float]]) -> float:
        raise NotImplementedError

    def _probabilities(self, sample_ids: Sequence[str], context: "SelectionContext") -> List[List[List[float]]]:
        return _normalize_probability_cube(
            context.predict_committee(sample_ids, batch_size=self.batch_size),
            sample_ids,
            strategy_name=self.name,
            method_name="predict_committee",
            min_member_count=2,
            expected_width=_label_schema_width(context),
        )

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> List[str]:
        sample_ids = _unique_pool_ids(pool_ids)
        target_k = _target_count(sample_ids, k)
        if target_k <= 0:
            return []
        cube = self._probabilities(sample_ids, context)
        return _select_by_cube_scores(self.name, sample_ids, target_k, context, cube, self.score)


class CommitteeVoteEntropyStrategy(_CommitteeStrategyBase):
    """Select samples with highest entropy of committee argmax votes."""

    name = "committee_vote_entropy"

    def score(self, probabilities: Sequence[Sequence[float]]) -> float:
        votes = [_argmax(row) for row in probabilities]
        counts = Counter(votes)
        distribution = [count / len(votes) for count in counts.values()]
        return _entropy(distribution)


class CommitteeKLDivergenceStrategy(_CommitteeStrategyBase):
    """Select samples with largest mean member KL divergence from consensus."""

    name = "committee_kl_divergence"

    def score(self, probabilities: Sequence[Sequence[float]]) -> float:
        consensus = _mean_probability(probabilities)
        total = 0.0
        for row in probabilities:
            for probability, consensus_probability in zip(row, consensus):
                if probability > 0.0:
                    total += probability * math.log(probability / consensus_probability)
        return total / len(probabilities)


class CommitteePairwiseDisagreementStrategy(_CommitteeStrategyBase):
    """Select samples with highest mean pairwise committee vote disagreement."""

    name = "committee_pairwise_disagreement"

    def score(self, probabilities: Sequence[Sequence[float]]) -> float:
        votes = [_argmax(row) for row in probabilities]
        if len(votes) < 2:
            return 0.0
        disagreements = 0
        pairs = 0
        for left_index in range(len(votes)):
            for right_index in range(left_index + 1, len(votes)):
                pairs += 1
                if votes[left_index] != votes[right_index]:
                    disagreements += 1
        return disagreements / pairs


class CommitteeMarginStrategy(_CommitteeStrategyBase):
    """Select samples with the smallest top-two committee vote margin."""

    name = "committee_margin"

    def score(self, probabilities: Sequence[Sequence[float]]) -> float:
        width = len(probabilities[0])
        votes = [_argmax(row) for row in probabilities]
        counts = [0 for _ in range(width)]
        for vote in votes:
            counts[vote] += 1
        ordered = sorted(counts, reverse=True)
        top = ordered[0] / len(votes)
        second = ordered[1] / len(votes) if len(ordered) > 1 else 0.0
        return 1.0 - (top - second)
