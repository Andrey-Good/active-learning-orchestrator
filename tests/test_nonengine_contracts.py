from __future__ import annotations

import json
from typing import Any

import pytest

from active_learning_sdk.annotation import AnnotationAggregator
from active_learning_sdk.backends.label_studio import LabelStudioBackend
from active_learning_sdk.configs import AnnotationPolicy, LabelBackendConfig, LabelSchema
from active_learning_sdk.dataset.provider import DataFrameDatasetProvider
from active_learning_sdk.exceptions import ConfigurationError
from active_learning_sdk.types import AnnotationRecord, SampleStatus


def _annotation(annotator_id: str, value: Any, created_at: float) -> AnnotationRecord:
    return AnnotationRecord(annotator_id=annotator_id, value=value, created_at=created_at)


@pytest.mark.parametrize("mode", ["latest", "first", "majority", "consensus"])
def test_annotation_policy_disallowing_single_annotator_requires_two_min_votes(mode: str) -> None:
    with pytest.raises(ConfigurationError, match="min_votes must be >= 2"):
        AnnotationPolicy(mode=mode, min_votes=1, allow_single_annotator=False).validate()


def test_annotation_policy_requires_distinct_annotators_when_single_annotator_is_disallowed() -> None:
    aggregator = AnnotationAggregator(
        AnnotationPolicy(mode="majority", min_votes=2, allow_single_annotator=False)
    )

    resolved = aggregator.resolve(
        "s1",
        [
            _annotation("same-user", "positive", 1.0),
            _annotation("same-user", "positive", 2.0),
        ],
    )

    assert resolved.status == SampleStatus.NEEDS_REVIEW
    assert resolved.details["reason"] == "distinct_min_votes_not_reached"
    assert resolved.details["eligible_votes"] == 1


def test_duplicate_annotator_votes_do_not_weight_majority_when_distinct_votes_are_required() -> None:
    aggregator = AnnotationAggregator(
        AnnotationPolicy(mode="majority", min_votes=2, min_agreement=0.67, allow_single_annotator=False)
    )

    resolved = aggregator.resolve(
        "s1",
        [
            _annotation("a1", "positive", 1.0),
            _annotation("a1", "positive", 2.0),
            _annotation("a2", "negative", 3.0),
        ],
    )

    assert resolved.status == SampleStatus.NEEDS_REVIEW
    assert resolved.label is None
    assert resolved.details["reason"] == "majority_tie"
    assert resolved.details["counts"] == {"positive": 1, "negative": 1}


def test_majority_tie_routes_to_review_instead_of_picking_arbitrary_label() -> None:
    aggregator = AnnotationAggregator(AnnotationPolicy(mode="majority", min_votes=2, min_agreement=0.5))

    resolved = aggregator.resolve(
        "s1",
        [
            _annotation("a1", "positive", 1.0),
            _annotation("a2", "negative", 2.0),
        ],
    )

    assert resolved.status == SampleStatus.NEEDS_REVIEW
    assert resolved.label is None
    assert resolved.details["reason"] == "majority_tie"


def test_label_studio_poll_requires_distinct_annotators_when_policy_disallows_single_annotator() -> None:
    class TaskClient:
        def request(self, method: str, path: str, **_: Any) -> dict[str, Any]:
            assert method == "GET"
            assert path == "/api/tasks/100/"
            return {
                "id": 100,
                "meta": {"sdk_round_id": "r1", "sdk_sample_id": "s1"},
                "annotations": [
                    {
                        "completed_by": {"email": "same@example.com"},
                        "result": [{"value": {"choices": ["positive"]}}],
                    },
                    {
                        "completed_by": {"email": "same@example.com"},
                        "result": [{"value": {"choices": ["positive"]}}],
                    },
                ],
            }

    backend = LabelStudioBackend(
        LabelBackendConfig(
            backend="label_studio",
            mode="external",
            url="http://label-studio.local",
            api_token="token",
            project_id=10,
        )
    )
    backend._ready = True
    backend._project_id = "10"
    backend._label_schema = LabelSchema(task="text_classification", labels=["negative", "positive"])
    backend._http_client = TaskClient()

    progress = backend.poll_round(
        "r1",
        {"s1": "100"},
        AnnotationPolicy(min_votes=2, allow_single_annotator=False),
    )

    assert progress.done == 0
    assert progress.ready_sample_ids == []
    assert progress.details["tasks"]["s1"]["eligible_votes"] == 1


def test_simulator_poll_requires_distinct_annotators_when_policy_disallows_single_annotator() -> None:
    from active_learning_sdk.backends.simulator import SimulatorLabelBackend
    from active_learning_sdk.types import DataSample

    backend = SimulatorLabelBackend(oracle_on="pull")
    backend.ensure_ready(LabelSchema(task="text_classification", labels=["negative", "positive"]))
    push = backend.push_round("r1", [DataSample(sample_id="s1", data={"text": "one"})])
    backend.submit_annotation(task_id=push.task_ids["s1"], value="positive", annotator_id="same-user", created_at=1.0)
    backend.submit_annotation(task_id=push.task_ids["s1"], value="positive", annotator_id="same-user", created_at=2.0)

    progress = backend.poll_round(
        "r1",
        {"s1": push.task_ids["s1"]},
        AnnotationPolicy(min_votes=2, allow_single_annotator=False),
    )

    assert progress.done == 0
    assert progress.ready_sample_ids == []
    assert progress.details["tasks"]["s1"]["annotations"] == 2
    assert progress.details["tasks"]["s1"]["eligible_votes"] == 1


def test_simulator_push_round_is_idempotent_only_for_same_sample_set() -> None:
    from active_learning_sdk.backends.simulator import SimulatorLabelBackend
    from active_learning_sdk.exceptions import LabelBackendError
    from active_learning_sdk.types import DataSample

    backend = SimulatorLabelBackend()
    backend.ensure_ready(LabelSchema(task="text_classification", labels=["negative", "positive"]))
    first = backend.push_round("r1", [DataSample(sample_id="s1", data={"text": "one"})])
    retry = backend.push_round("r1", [DataSample(sample_id="s1", data={"text": "one again"})])

    assert retry.task_ids == first.task_ids
    with pytest.raises(LabelBackendError, match="already exists with sample ids"):
        backend.push_round("r1", [DataSample(sample_id="s2", data={"text": "two"})])


@pytest.mark.parametrize(
    "labels",
    [
        [1],
        [""],
        ["  "],
        ["positive", "positive"],
        [["positive"]],
    ],
)
def test_label_schema_rejects_invalid_labels(labels: list[Any]) -> None:
    with pytest.raises(ConfigurationError):
        LabelSchema(task="text_classification", labels=labels).validate()


def test_dataframe_provider_normalizes_extra_values_to_strict_json_safe_values() -> None:
    pd = pytest.importorskip("pandas")
    dataset = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "text": "hello",
                "score": float("nan"),
                "count": pd.Series([3], dtype="int64")[0],
                "created": pd.Timestamp("2026-04-27T01:02:03"),
                "meta": {"reviewed": pd.NA, "rank": pd.Series([2], dtype="int64")[0]},
                "group_id": pd.NA,
            }
        ]
    )

    sample = DataFrameDatasetProvider(dataset).get_sample("s1")

    assert sample.data["score"] is None
    assert sample.data["count"] == 3
    assert sample.data["created"] == "2026-04-27T01:02:03"
    assert sample.meta == {"reviewed": None, "rank": 2}
    assert sample.group_id is None
    json.dumps({"data": sample.data, "meta": sample.meta, "group_id": sample.group_id}, allow_nan=False)


@pytest.mark.parametrize("text_value", [None, float("nan"), ["not", "text"], {"not": "text"}, 42])
def test_dataframe_provider_rejects_invalid_text_values_before_stringifying(text_value: Any) -> None:
    pd = pytest.importorskip("pandas")
    dataset = pd.DataFrame([{"sample_id": "s1", "text": text_value}])
    provider = DataFrameDatasetProvider(dataset)

    with pytest.raises(ConfigurationError, match="text column.*string|text column.*missing"):
        provider.get_sample("s1")


@pytest.mark.parametrize("sample_id", [None, float("nan"), 1, 1.5, ["s1"], {"id": "s1"}, ""])
def test_dataframe_provider_rejects_invalid_sample_ids_before_coercion(sample_id: Any) -> None:
    pd = pytest.importorskip("pandas")
    dataset = pd.DataFrame([{"sample_id": sample_id, "text": "hello"}])

    with pytest.raises(ConfigurationError, match="sample_id column"):
        DataFrameDatasetProvider(dataset)
