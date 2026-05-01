"""
Annotation aggregation (turn many annotations into one label).

Backends can return multiple annotations per sample (for example, multiple humans
in Label Studio). The engine cannot train on "a list of annotations", it needs a
single final label per sample. This module contains that conversion logic.
"""

from __future__ import annotations


from typing import Any, Dict, Sequence

from .configs import AnnotationPolicy
from .exceptions import ConfigurationError
from .types import AnnotationRecord, ResolvedLabel, SampleStatus


class AnnotationAggregator:
    """
    Aggregate multiple annotations into a single resolved label.

    For juniors:
    - You usually do NOT change engine logic to add a new aggregation mode.
    - Instead, add a new mode to `AnnotationPolicy` and implement it here.

    Attributes:
        policy (AnnotationPolicy):
            Where: set in `__init__` and used in `resolve()` to pick aggregation rules.
            What: aggregation configuration (mode, min_votes, min_agreement).
            Why: allows changing aggregation behavior without rewriting engine steps.
    """

    def __init__(self, policy: AnnotationPolicy) -> None:
        policy.validate()
        self.policy = policy

    def resolve(self, sample_id: str, annotations: Sequence[AnnotationRecord]) -> ResolvedLabel:
        """
        Resolve annotations for one sample into a `ResolvedLabel`.

        Inputs:
        - `sample_id`: the dataset id (stable across runs)
        - `annotations`: a list of AnnotationRecord from the backend

        Output:
        - status LABELED: engine writes the label to state and uses it for training
        - status NEEDS_REVIEW: engine marks the sample as "needs_review"
        - status UNLABELED: engine keeps the sample unlabeled (rare in this implementation)
        """
        if not annotations:
            return ResolvedLabel(
                sample_id=sample_id,
                status=SampleStatus.UNLABELED,
                details={"reason": "no_annotations", "votes": 0, "eligible_votes": 0},
            )

        sorted_annotations = sorted(list(annotations), key=lambda item: item.created_at)
        eligible_annotations = self._eligible_annotations(sorted_annotations)
        vote_count = len(eligible_annotations)
        if vote_count < self.policy.min_votes:
            reason = "min_votes_not_reached" if self.policy.allow_single_annotator else "distinct_min_votes_not_reached"
            return ResolvedLabel(
                sample_id=sample_id,
                status=SampleStatus.NEEDS_REVIEW,
                details={
                    "reason": reason,
                    "votes": len(sorted_annotations),
                    "eligible_votes": vote_count,
                },
            )

        if self.policy.mode == "latest":
            return ResolvedLabel(sample_id=sample_id, status=SampleStatus.LABELED, label=eligible_annotations[-1].value)

        if self.policy.mode == "first":
            return ResolvedLabel(sample_id=sample_id, status=SampleStatus.LABELED, label=eligible_annotations[0].value)

        if self.policy.mode in {"majority", "consensus"}:
            normalized_values = [self._normalize_label(item.value) for item in eligible_annotations]
            counts: Dict[Any, int] = {}
            for value in normalized_values:
                counts[value] = counts.get(value, 0) + 1
            best_label, best_votes = max(counts.items(), key=lambda entry: entry[1])
            agreement = best_votes / max(1, len(normalized_values))
            tied_labels = [label for label, votes in counts.items() if votes == best_votes]

            if self.policy.mode == "consensus" and len(counts) > 1:
                return ResolvedLabel(
                    sample_id=sample_id,
                    status=SampleStatus.NEEDS_REVIEW,
                    agreement=agreement,
                    details={"reason": "no_consensus", "counts": counts},
                )

            if self.policy.mode == "majority" and len(tied_labels) > 1:
                return ResolvedLabel(
                    sample_id=sample_id,
                    status=SampleStatus.NEEDS_REVIEW,
                    agreement=agreement,
                    details={"reason": "majority_tie", "counts": counts},
                )

            if agreement < self.policy.min_agreement:
                return ResolvedLabel(
                    sample_id=sample_id,
                    status=SampleStatus.NEEDS_REVIEW,
                    agreement=agreement,
                    details={"reason": "min_agreement_not_reached", "counts": counts},
                )

            return ResolvedLabel(
                sample_id=sample_id,
                status=SampleStatus.LABELED,
                label=self._denormalize_label(best_label),
                agreement=agreement,
                details={"counts": counts},
            )

        raise ConfigurationError(f"Unsupported annotation aggregation mode: {self.policy.mode!r}")

    def _eligible_annotations(self, annotations: Sequence[AnnotationRecord]) -> list[AnnotationRecord]:
        if self.policy.allow_single_annotator:
            return list(annotations)
        latest_by_annotator: Dict[str, AnnotationRecord] = {}
        for annotation in annotations:
            latest_by_annotator[str(annotation.annotator_id)] = annotation
        return sorted(latest_by_annotator.values(), key=lambda item: item.created_at)

    def _normalize_label(self, value: Any) -> Any:
        """Convert values into a hashable form for vote counting."""
        if isinstance(value, list):
            return tuple(sorted(value))
        return value

    def _denormalize_label(self, value: Any) -> Any:
        """Convert normalized labels back to the user-facing format."""
        if isinstance(value, tuple):
            return list(value)
        return value
