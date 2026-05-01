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
    SchedulerConfig,
    SplitConfig,
)
from active_learning_sdk.adapters.base import (
    TextClassificationAdapter,
    inspect_model_capabilities,
    unsupported_adapter_method,
)
from active_learning_sdk.backends.base import RoundProgress, RoundPullResult, RoundPushResult
from active_learning_sdk.types import DataSample


class InMemoryDataset:
    def __init__(self) -> None:
        self._samples = {
            "s1": DataSample(sample_id="s1", data={"text": "one"}),
            "s2": DataSample(sample_id="s2", data={"text": "two"}),
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

    def close(self) -> None:
        return None


class FitEvaluateOnlyModel:
    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        return {"accuracy": 0.0}


class ProtocolFitEvaluateOnlyModel(TextClassificationAdapter):
    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        return {"accuracy": 0.0}


class ProbabilityModel(FitEvaluateOnlyModel):
    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [[0.5, 0.5] for _ in texts]


class EmbeddingModel(ProbabilityModel):
    def embed(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [[0.0, 1.0] for _ in texts]


class OptionalCapabilityModel(EmbeddingModel):
    def predict_logits(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [[0.0, 0.0] for _ in texts]

    def gradient_embed(
        self,
        texts: Sequence[str],
        labels: Sequence[Any] | None = None,
        batch_size: int = 32,
    ) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def predict_stochastic(self, texts: Sequence[str], n: int = 10, batch_size: int = 32) -> list[list[list[float]]]:
        return [[[0.5, 0.5] for _ in range(n)] for _ in texts]

    def predict_committee(self, texts: Sequence[str], batch_size: int = 32) -> list[list[list[float]]]:
        return [[[0.5, 0.5], [0.5, 0.5]] for _ in texts]

    def get_model_id(self) -> str:
        return "optional-capability-model"

    def save(self, path: str) -> None:
        return None

    def load(self, path: str) -> None:
        return None


class UnsupportedPlaceholderModel(FitEvaluateOnlyModel):
    @unsupported_adapter_method("logits are not wired yet")
    def predict_logits(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        raise NotImplementedError

    @unsupported_adapter_method("gradient embeddings need autograd")
    def gradient_embed(
        self,
        texts: Sequence[str],
        labels: Sequence[Any] | None = None,
        batch_size: int = 32,
    ) -> list[list[float]]:
        raise NotImplementedError


class CustomProbabilityStrategy:
    name = "custom_probability"
    required_capabilities = frozenset({"predict_proba"})

    def select(self, pool_ids: Sequence[str], k: int, context: Any) -> list[str]:
        return list(pool_ids)[:k]


def _configure_project(
    tmp_path: Path,
    *,
    model: Any,
    scheduler_config: SchedulerConfig,
    strategies: Sequence[Any] | None = None,
) -> ActiveLearningProject:
    project = ActiveLearningProject("capability-test", tmp_path, lock=False)
    project.configure(
        dataset=InMemoryDataset(),
        model=model,
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=NoopBackend(),
        scheduler_config=scheduler_config,
        cache_config=CacheConfig(enable=False),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["s1"], "val": ["s2"], "test": []},
        ),
        strategies=strategies,
    )
    return project


def test_random_configures_without_predict_proba(tmp_path: Path) -> None:
    project = _configure_project(
        tmp_path,
        model=FitEvaluateOnlyModel(),
        scheduler_config=SchedulerConfig(strategy="random"),
    )

    assert project.get_state().scheduler_config["strategy"] == "random"


def test_entropy_fails_configure_without_predict_proba(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="entropy.*predict_proba"):
        _configure_project(
            tmp_path,
            model=FitEvaluateOnlyModel(),
            scheduler_config=SchedulerConfig(strategy="entropy"),
        )


def test_entropy_configures_without_predict_proba_when_capabilities_are_non_strict(tmp_path: Path) -> None:
    project = _configure_project(
        tmp_path,
        model=FitEvaluateOnlyModel(),
        scheduler_config=SchedulerConfig(strategy="entropy", strict_capabilities=False),
    )

    assert project.get_state().scheduler_config["strategy"] == "entropy"
    assert project.get_state().scheduler_config["strict_capabilities"] is False


def test_entropy_fails_when_protocol_predict_proba_is_not_implemented(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="entropy.*predict_proba.*not implemented on adapter"):
        _configure_project(
            tmp_path,
            model=ProtocolFitEvaluateOnlyModel(),
            scheduler_config=SchedulerConfig(strategy="entropy"),
        )


def test_entropy_fails_attach_runtime_without_predict_proba(tmp_path: Path) -> None:
    project = _configure_project(
        tmp_path,
        model=ProbabilityModel(),
        scheduler_config=SchedulerConfig(strategy="entropy"),
    )
    project.close()
    reopened = ActiveLearningProject("capability-test", tmp_path, lock=False)

    with pytest.raises(ConfigurationError, match="entropy.*predict_proba"):
        reopened.attach_runtime(
            dataset=InMemoryDataset(),
            model=FitEvaluateOnlyModel(),
            label_backend=NoopBackend(),
        )


def test_entropy_attach_runtime_allows_missing_predict_proba_when_capabilities_are_non_strict(tmp_path: Path) -> None:
    project = _configure_project(
        tmp_path,
        model=ProbabilityModel(),
        scheduler_config=SchedulerConfig(strategy="entropy", strict_capabilities=False),
    )
    project.close()
    reopened = ActiveLearningProject("capability-test", tmp_path, lock=False)

    reopened.attach_runtime(
        dataset=InMemoryDataset(),
        model=FitEvaluateOnlyModel(),
        label_backend=NoopBackend(),
    )

    assert reopened.get_state().scheduler_config["strict_capabilities"] is False


def test_non_strict_entropy_fails_at_selection_without_predict_proba(tmp_path: Path) -> None:
    project = _configure_project(
        tmp_path,
        model=FitEvaluateOnlyModel(),
        scheduler_config=SchedulerConfig(strategy="entropy", strict_capabilities=False),
    )

    with pytest.raises(ConfigurationError, match="entropy.*predict_proba"):
        project.run_step(batch_size=1)


def test_mix_interleaved_fails_when_any_arm_lacks_required_capability(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="entropy.*predict_proba"):
        _configure_project(
            tmp_path,
            model=FitEvaluateOnlyModel(),
            scheduler_config=SchedulerConfig(mode="mix_interleaved", mix={"random": 0.5, "entropy": 0.5}),
        )


def test_non_strict_mix_interleaved_configures_with_missing_arm_capability(tmp_path: Path) -> None:
    project = _configure_project(
        tmp_path,
        model=FitEvaluateOnlyModel(),
        scheduler_config=SchedulerConfig(
            mode="mix_interleaved",
            mix={"random": 0.5, "entropy": 0.5},
            strict_capabilities=False,
        ),
    )

    assert project.get_state().scheduler_config["mode"] == "mix_interleaved"


def test_non_strict_custom_strategy_fails_at_selection_without_required_capability(tmp_path: Path) -> None:
    project = _configure_project(
        tmp_path,
        model=FitEvaluateOnlyModel(),
        scheduler_config=SchedulerConfig(strategy="custom_probability", strict_capabilities=False),
    )
    project.register_strategy(CustomProbabilityStrategy())

    with pytest.raises(ConfigurationError, match="custom_probability.*predict_proba"):
        project.run_step(batch_size=1)


def test_strict_custom_strategy_configure_accepts_supplied_strategy_with_required_capability(tmp_path: Path) -> None:
    project = _configure_project(
        tmp_path,
        model=ProbabilityModel(),
        scheduler_config=SchedulerConfig(strategy="custom_probability"),
        strategies=[CustomProbabilityStrategy()],
    )

    assert project.get_state().scheduler_config["strategy"] == "custom_probability"


def test_strict_custom_strategy_configure_rejects_supplied_strategy_without_capability(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="custom_probability.*predict_proba"):
        _configure_project(
            tmp_path,
            model=FitEvaluateOnlyModel(),
            scheduler_config=SchedulerConfig(strategy="custom_probability"),
            strategies=[CustomProbabilityStrategy()],
        )


def test_strict_custom_strategy_attach_runtime_accepts_supplied_strategy(tmp_path: Path) -> None:
    project = _configure_project(
        tmp_path,
        model=ProbabilityModel(),
        scheduler_config=SchedulerConfig(strategy="custom_probability"),
        strategies=[CustomProbabilityStrategy()],
    )
    project.close()
    reopened = ActiveLearningProject("capability-test", tmp_path, lock=False)

    reopened.attach_runtime(
        dataset=InMemoryDataset(),
        model=ProbabilityModel(),
        label_backend=NoopBackend(),
        strategies=[CustomProbabilityStrategy()],
    )

    assert reopened.get_state().scheduler_config["strategy"] == "custom_probability"


def test_register_strategy_validates_active_strict_strategy_before_overriding(tmp_path: Path) -> None:
    class ProbabilityHungryRandom:
        name = "random"
        required_capabilities = frozenset({"predict_proba"})

        def select(self, pool_ids: Sequence[str], k: int, context: Any) -> list[str]:
            return list(pool_ids)[:k]

    project = _configure_project(
        tmp_path,
        model=FitEvaluateOnlyModel(),
        scheduler_config=SchedulerConfig(strategy="random"),
    )

    with pytest.raises(ConfigurationError, match="random.*predict_proba"):
        project.register_strategy(ProbabilityHungryRandom())


def test_coreset_kcenter_fails_configure_without_embed(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="coreset_kcenter.*embed"):
        _configure_project(
            tmp_path,
            model=ProbabilityModel(),
            scheduler_config=SchedulerConfig(strategy="coreset_kcenter"),
        )


def test_coreset_kcenter_configures_with_embed(tmp_path: Path) -> None:
    project = _configure_project(
        tmp_path,
        model=EmbeddingModel(),
        scheduler_config=SchedulerConfig(strategy="coreset_kcenter"),
    )

    assert project.get_state().scheduler_config["strategy"] == "coreset_kcenter"


def test_embedding_strategies_configure_with_embed(tmp_path: Path) -> None:
    for strategy_name in (
        "adaptive_uncertainty_diversity",
        "embedding_kmeans_pp",
        "max_min_embedding",
        "deduplicate_near_neighbors",
        "density_weighted_diversity",
    ):
        workdir = tmp_path / strategy_name
        _configure_project(
            workdir,
            model=EmbeddingModel(),
            scheduler_config=SchedulerConfig(strategy=strategy_name),
        )


def test_stochastic_strategy_fails_configure_without_predict_stochastic(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="mc_dropout_entropy.*predict_stochastic"):
        _configure_project(
            tmp_path,
            model=ProbabilityModel(),
            scheduler_config=SchedulerConfig(strategy="mc_dropout_entropy"),
        )


def test_stochastic_strategy_fails_attach_runtime_without_predict_stochastic(tmp_path: Path) -> None:
    project = _configure_project(
        tmp_path,
        model=OptionalCapabilityModel(),
        scheduler_config=SchedulerConfig(strategy="mc_dropout_entropy"),
    )
    project.close()
    reopened = ActiveLearningProject("capability-test", tmp_path, lock=False)

    with pytest.raises(ConfigurationError, match="mc_dropout_entropy.*predict_stochastic"):
        reopened.attach_runtime(
            dataset=InMemoryDataset(),
            model=ProbabilityModel(),
            label_backend=NoopBackend(),
        )


def test_committee_strategy_fails_configure_without_predict_committee(tmp_path: Path) -> None:
    class StochasticOnlyModel(ProbabilityModel):
        def predict_stochastic(self, texts: Sequence[str], n: int = 10, batch_size: int = 32) -> list[list[list[float]]]:
            return [[[0.5, 0.5] for _ in range(n)] for _ in texts]

    with pytest.raises(ConfigurationError, match="committee_vote_entropy.*predict_committee"):
        _configure_project(
            tmp_path,
            model=StochasticOnlyModel(),
            scheduler_config=SchedulerConfig(strategy="committee_vote_entropy"),
        )


def test_inspect_model_capabilities_recognizes_optional_methods_and_unsupported_placeholders() -> None:
    full_caps = inspect_model_capabilities(OptionalCapabilityModel())

    assert full_caps.predict_proba is True
    assert full_caps.predict_logits is True
    assert full_caps.embed is True
    assert full_caps.gradient_embed is True
    assert full_caps.predict_stochastic is True
    assert full_caps.predict_committee is True
    assert full_caps.fit is True
    assert full_caps.evaluate is True
    assert full_caps.get_model_id is True
    assert full_caps.save_load is True

    placeholder_caps = inspect_model_capabilities(UnsupportedPlaceholderModel())

    assert placeholder_caps.predict_logits is False
    assert placeholder_caps.gradient_embed is False
    assert placeholder_caps.unsupported_methods["predict_logits"] == "logits are not wired yet"
    assert placeholder_caps.unsupported_methods["gradient_embed"] == "gradient embeddings need autograd"


def test_inspect_model_capabilities_rejects_protocol_stubs() -> None:
    caps = inspect_model_capabilities(ProtocolFitEvaluateOnlyModel())

    assert caps.predict_proba is False
    assert caps.predict_logits is False
    assert caps.embed is False
    assert caps.gradient_embed is False
    assert caps.predict_stochastic is False
    assert caps.predict_committee is False
    assert caps.fit is True
    assert caps.evaluate is True
    assert caps.get_model_id is False
    assert caps.save_load is False
    assert caps.unsupported_methods["predict_proba"] == "not implemented on adapter"
    assert caps.unsupported_methods["predict_logits"] == "not implemented on adapter"
    assert caps.unsupported_methods["embed"] == "not implemented on adapter"
    assert caps.unsupported_methods["gradient_embed"] == "not implemented on adapter"
    assert caps.unsupported_methods["predict_stochastic"] == "not implemented on adapter"
    assert caps.unsupported_methods["predict_committee"] == "not implemented on adapter"
    assert caps.unsupported_methods["get_model_id"] == "not implemented on adapter"
    assert caps.unsupported_methods["save"] == "not implemented on adapter"
    assert caps.unsupported_methods["load"] == "not implemented on adapter"


def test_inspect_model_capabilities_treats_hostile_optional_properties_as_unsupported() -> None:
    class HostilePropertyModel(FitEvaluateOnlyModel):
        @property
        def predict_proba(self) -> Any:
            raise RuntimeError("property exploded")

    caps = inspect_model_capabilities(HostilePropertyModel())

    assert caps.predict_proba is False
    assert "attribute access failed: RuntimeError: property exploded" == caps.unsupported_methods["predict_proba"]
    assert caps.fit is True
    assert caps.evaluate is True
