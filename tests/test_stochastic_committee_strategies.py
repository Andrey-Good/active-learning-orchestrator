from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from active_learning_sdk import (
    ActiveLearningProject,
    CacheConfig,
    ConfigurationError,
    LabelBackendConfig,
    LabelSchema,
    ModelAdapterError,
    SchedulerConfig,
    SplitConfig,
)
from active_learning_sdk.backends.base import RoundProgress, RoundPullResult, RoundPushResult
from active_learning_sdk.engine import SelectionContext, StrategyScheduler
from active_learning_sdk.strategies import (
    BaldStrategy,
    CommitteeKLDivergenceStrategy,
    CommitteeMarginStrategy,
    CommitteePairwiseDisagreementStrategy,
    CommitteeVoteEntropyStrategy,
    McDropoutEntropyStrategy,
    PredictionVarianceStrategy,
    VariationRatioStrategy,
)
from active_learning_sdk.types import DataSample


class FakeStochasticContext:
    def __init__(self, predictions: Mapping[str, Any]) -> None:
        self.predictions = dict(predictions)
        self.calls: list[tuple[list[str], int, int]] = []

    def model_id(self) -> str:
        return "fake-stochastic"

    def predict_stochastic(self, sample_ids: Sequence[str], n: int = 10, batch_size: int = 32) -> list[Any]:
        ids = [str(sample_id) for sample_id in sample_ids]
        self.calls.append((ids, n, batch_size))
        return [self.predictions[sample_id] for sample_id in ids]


class FakeCommitteeContext:
    def __init__(self, predictions: Mapping[str, Any]) -> None:
        self.predictions = dict(predictions)
        self.calls: list[tuple[list[str], int]] = []

    def model_id(self) -> str:
        return "fake-committee"

    def predict_committee(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[Any]:
        ids = [str(sample_id) for sample_id in sample_ids]
        self.calls.append((ids, batch_size))
        return [self.predictions[sample_id] for sample_id in ids]


class InMemoryDataset:
    def __init__(self) -> None:
        self._samples = {
            "s1": DataSample(sample_id="s1", data={"text": "one"}),
            "s2": DataSample(sample_id="s2", data={"text": "two"}),
            "s3": DataSample(sample_id="s3", data={"text": "three"}),
        }

    def iter_sample_ids(self):
        yield from self._samples

    def get_sample(self, sample_id: str) -> DataSample:
        return self._samples[sample_id]

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        return [self.get_sample(sample_id) for sample_id in sample_ids]

    def schema(self) -> dict[str, str]:
        return {"sample_id": "str", "text": "str"}


class NoopBackend:
    def ensure_ready(self, label_schema: LabelSchema) -> dict[str, Any]:
        return {"backend": "noop"}

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: dict[str, Any] | None = None,
    ) -> RoundPushResult:
        return RoundPushResult(task_ids={sample.sample_id: sample.sample_id for sample in samples})

    def poll_round(
        self,
        round_id: str,
        task_ids: Mapping[str, str],
        policy: Any,
    ) -> RoundProgress:
        return RoundProgress(total=len(task_ids), done=len(task_ids), ready_sample_ids=list(task_ids))

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        return RoundPullResult(annotations={})


class FitEvaluateModel:
    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        return {"accuracy": 0.0}


class FullUncertaintyModel(FitEvaluateModel):
    def predict_stochastic(self, texts: Sequence[str], n: int = 10, batch_size: int = 32) -> list[list[list[float]]]:
        return [[[0.6, 0.4] for _ in range(n)] for _ in texts]

    def predict_committee(self, texts: Sequence[str], batch_size: int = 32) -> list[list[list[float]]]:
        return [[[0.6, 0.4], [0.4, 0.6]] for _ in texts]


STOCHASTIC_PREDICTIONS = {
    "stable": [[0.9, 0.1] for _ in range(10)],
    "mean_entropy": [[0.5, 0.5] for _ in range(10)],
    "disagreement": [[0.99, 0.01] if index < 6 else [0.01, 0.99] for index in range(10)],
}


COMMITTEE_PREDICTIONS = {
    "stable": [[0.9, 0.1, 0.0], [0.85, 0.15, 0.0], [0.95, 0.05, 0.0]],
    "partial": [[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.0, 0.2, 0.8]],
    "spread": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
}


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (McDropoutEntropyStrategy(), "mean_entropy"),
        (BaldStrategy(), "disagreement"),
        (VariationRatioStrategy(), "disagreement"),
        (PredictionVarianceStrategy(), "disagreement"),
    ],
)
def test_stochastic_strategies_select_expected_highest_scoring_sample(strategy: Any, expected: str) -> None:
    context = FakeStochasticContext(STOCHASTIC_PREDICTIONS)

    selected = strategy.select(["stable", "mean_entropy", "disagreement"], 1, context)

    assert selected == [expected]
    assert context.calls == [(["stable", "mean_entropy", "disagreement"], 10, 32)]


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (CommitteeVoteEntropyStrategy(), "spread"),
        (CommitteeKLDivergenceStrategy(), "spread"),
        (CommitteePairwiseDisagreementStrategy(), "spread"),
        (CommitteeMarginStrategy(), "spread"),
    ],
)
def test_committee_strategies_select_expected_highest_scoring_sample(strategy: Any, expected: str) -> None:
    context = FakeCommitteeContext(COMMITTEE_PREDICTIONS)

    selected = strategy.select(["stable", "partial", "spread"], 1, context)

    assert selected == [expected]
    assert context.calls == [(["stable", "partial", "spread"], 32)]


def test_stochastic_strategies_handle_empty_k_overflow_and_duplicate_pool_ids() -> None:
    context = FakeStochasticContext({"a": [[0.5, 0.5] for _ in range(10)], "b": [[0.9, 0.1] for _ in range(10)]})
    strategy = McDropoutEntropyStrategy()

    assert strategy.select([], 3, context) == []
    assert strategy.select(["a"], 0, context) == []
    assert strategy.select(["a"], -1, context) == []
    assert strategy.select(["b", "a", "b", "a"], 10, context) == ["a", "b"]


@pytest.mark.parametrize("pass_count", [9, 11])
def test_stochastic_output_with_wrong_requested_pass_count_raises_configuration_error(pass_count: int) -> None:
    context = FakeStochasticContext({"a": [[0.5, 0.5] for _ in range(pass_count)]})

    with pytest.raises(ConfigurationError, match=rf"mc_dropout_entropy\.predict_stochastic.*has {pass_count}.*expected 10"):
        McDropoutEntropyStrategy().select(["a"], 1, context)


def test_stochastic_output_with_inconsistent_pass_counts_raises_configuration_error() -> None:
    context = FakeStochasticContext(
        {
            "a": [[0.5, 0.5] for _ in range(10)],
            "b": [[0.5, 0.5] for _ in range(9)],
        }
    )

    with pytest.raises(ConfigurationError, match=r"mc_dropout_entropy\.predict_stochastic.*sample 'b'.*has 9.*expected 10"):
        McDropoutEntropyStrategy().select(["a", "b"], 1, context)


def test_committee_output_with_inconsistent_member_counts_raises_configuration_error() -> None:
    context = FakeCommitteeContext(
        {
            "a": [[0.5, 0.5], [0.6, 0.4]],
            "b": [[0.5, 0.5]],
        }
    )

    with pytest.raises(
        ConfigurationError,
        match=r"committee_vote_entropy\.predict_committee.*sample 'b'.*has 1.*expected at least 2",
    ):
        CommitteeVoteEntropyStrategy().select(["a", "b"], 1, context)


@pytest.mark.parametrize(
    "strategy",
    [
        CommitteeVoteEntropyStrategy(),
        CommitteeKLDivergenceStrategy(),
        CommitteePairwiseDisagreementStrategy(),
        CommitteeMarginStrategy(),
    ],
)
def test_committee_strategies_reject_one_member_committees(strategy: Any) -> None:
    context = FakeCommitteeContext({"a": [[0.5, 0.5]]})

    with pytest.raises(
        ConfigurationError,
        match=rf"{strategy.name}\.predict_committee.*sample 'a'.*has 1.*expected at least 2",
    ):
        strategy.select(["a"], 1, context)


def test_probability_row_sum_not_close_to_one_raises_configuration_error() -> None:
    context = FakeCommitteeContext({"a": [[0.6, 0.6], [0.5, 0.5]]})

    with pytest.raises(ConfigurationError, match=r"committee_vote_entropy\.predict_committee.*sums to 1.2.*expected 1.0"):
        CommitteeVoteEntropyStrategy().select(["a"], 1, context)


@pytest.mark.parametrize(
    ("predictions", "match"),
    [
        ([[[0.5, 0.5]]], "returned 1 rows for 2 sample ids"),
        ([[0.5, 0.5], [[0.5, 0.5]]], "has 2 passes/members; expected 1"),
        ([[], [[0.5, 0.5]]], "at least one pass/member"),
        ([[[]], [[0.5, 0.5]]], "must not be empty"),
        ([[[True, 0.5]], [[0.5, 0.5]]], "must be numeric"),
        ([[[float("nan"), 0.5]], [[0.5, 0.5]]], "must be finite"),
        ([[[-0.1, 1.1]], [[0.5, 0.5]]], "must be non-negative"),
        ([[[0.0, 0.0]], [[0.5, 0.5]]], "positive sum"),
        ([[[0.5, 0.5]], [[0.2, 0.3, 0.5]]], "expected 2"),
    ],
)
def test_stochastic_shape_validation_failures_raise_configuration_error(predictions: Any, match: str) -> None:
    class OnePassMcDropoutEntropyStrategy(McDropoutEntropyStrategy):
        n_passes = 1

    class MalformedContext(FakeStochasticContext):
        def predict_stochastic(self, sample_ids: Sequence[str], n: int = 1, batch_size: int = 32) -> Any:
            return predictions

    with pytest.raises(ConfigurationError, match=match):
        OnePassMcDropoutEntropyStrategy().select(["a", "b"], 1, MalformedContext({}))


@pytest.mark.parametrize(
    ("predictions", "match"),
    [
        ([[[0.5, 0.5]]], "returned 1 rows for 2 sample ids"),
        ([[0.5, 0.5], [[0.5, 0.5]]], "probability row 0:0.*must be a sequence"),
        ([[], [[0.5, 0.5]]], "at least one pass/member"),
        (
            [[[0.5, 0.5], [0.5, 0.5]], [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]],
            "expected 2",
        ),
        ([[[0.6, 0.6], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]], "sums to 1.2; expected 1.0"),
    ],
)
def test_committee_shape_validation_failures_raise_configuration_error(predictions: Any, match: str) -> None:
    class MalformedContext(FakeCommitteeContext):
        def predict_committee(self, sample_ids: Sequence[str], batch_size: int = 32) -> Any:
            return predictions

    with pytest.raises(ConfigurationError, match=match):
        CommitteeVoteEntropyStrategy().select(["a", "b"], 1, MalformedContext({}))


def test_scheduler_can_select_with_stochastic_and_committee_strategy_names() -> None:
    stochastic_selected, stochastic_snapshot = StrategyScheduler(
        SchedulerConfig(strategy="bald")
    ).select_batch(["stable", "mean_entropy", "disagreement"], 1, FakeStochasticContext(STOCHASTIC_PREDICTIONS), state={})

    committee_selected, committee_snapshot = StrategyScheduler(
        SchedulerConfig(strategy="committee_vote_entropy")
    ).select_batch(["stable", "partial", "spread"], 1, FakeCommitteeContext(COMMITTEE_PREDICTIONS), state={})

    assert stochastic_selected == ["disagreement"]
    assert stochastic_snapshot == {"mode": "single", "strategy": "bald"}
    assert committee_selected == ["spread"]
    assert committee_snapshot == {"mode": "single", "strategy": "committee_vote_entropy"}


def _selection_context(model: Any) -> SelectionContext:
    return SelectionContext(
        provider=InMemoryDataset(),
        model=model,
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        prediction_cache=None,
        embedding_cache=None,
        labeled_ids=[],
        last_metrics={},
    )


def test_selection_context_missing_capabilities_raise_configuration_error() -> None:
    context = _selection_context(FitEvaluateModel())

    with pytest.raises(ConfigurationError, match="predict_stochastic"):
        context.predict_stochastic(["s1"])
    with pytest.raises(ConfigurationError, match="predict_committee"):
        context.predict_committee(["s1"])


def test_selection_context_adapter_failures_are_wrapped() -> None:
    class FailingModel(FullUncertaintyModel):
        def predict_stochastic(self, texts: Sequence[str], n: int = 10, batch_size: int = 32) -> Any:
            raise RuntimeError("stochastic exploded")

        def predict_committee(self, texts: Sequence[str], batch_size: int = 32) -> Any:
            raise RuntimeError("committee exploded")

    context = _selection_context(FailingModel())

    with pytest.raises(ModelAdapterError, match="model.predict_stochastic failed"):
        context.predict_stochastic(["s1"])
    with pytest.raises(ModelAdapterError, match="model.predict_committee failed"):
        context.predict_committee(["s1"])


def test_selection_context_rejects_draw_major_stochastic_output() -> None:
    class DrawMajorModel(FullUncertaintyModel):
        def predict_stochastic(self, texts: Sequence[str], n: int = 10, batch_size: int = 32) -> Any:
            del batch_size
            return [[[0.6, 0.4] for _ in texts] for _ in range(n)]

    context = _selection_context(DrawMajorModel())

    with pytest.raises(ConfigurationError, match="model\\.predict_stochastic returned 10 rows for 2 sample ids"):
        context.predict_stochastic(["s1", "s2"], n=10)


def test_selection_context_rejects_member_major_committee_output() -> None:
    class MemberMajorModel(FullUncertaintyModel):
        def predict_committee(self, texts: Sequence[str], batch_size: int = 32) -> Any:
            del batch_size
            return [[[0.6, 0.4] for _ in texts] for _ in range(3)]

    context = _selection_context(MemberMajorModel())

    with pytest.raises(ConfigurationError, match="model\\.predict_committee returned 3 rows for 2 sample ids"):
        context.predict_committee(["s1", "s2"])


def test_configure_succeeds_for_stochastic_and_committee_strategies(tmp_path: Path) -> None:
    for strategy_name in ("mc_dropout_entropy", "committee_vote_entropy"):
        project = ActiveLearningProject(f"{strategy_name}-test", tmp_path / strategy_name, lock=False)
        project.configure(
            dataset=InMemoryDataset(),
            model=FullUncertaintyModel(),
            label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
            label_backend_config=LabelBackendConfig(backend="custom"),
            label_backend=NoopBackend(),
            scheduler_config=SchedulerConfig(strategy=strategy_name),
            cache_config=CacheConfig(enable=False),
            split_config=SplitConfig(
                mode="explicit",
                explicit_splits={"train": ["s1"], "val": ["s2"], "test": ["s3"]},
            ),
        )
        assert project.get_state().scheduler_config["strategy"] == strategy_name
