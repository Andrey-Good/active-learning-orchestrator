# Active Learning SDK

[![CI](https://github.com/Andrey-Good/active-learning-orchestrator/actions/workflows/ci.yml/badge.svg)](https://github.com/Andrey-Good/active-learning-orchestrator/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[English](#english) | [Русский](#russian)

<a id="english"></a>

## English

`active-learning-sdk` is a Python SDK for building resumable active learning loops for text classification. It keeps project state on disk, selects the next samples to label, sends them to a labeling backend, pulls annotations back, retrains your model adapter, and records reproducible reports.

The project is designed for teams that want active learning as a reliable workflow component rather than a pile of notebook glue code.

### What It Does

- Runs stateful active learning rounds: select, push, wait, pull, train/evaluate, update.
- Resumes interrupted projects without duplicating labeling tasks.
- Protects projects with dataset fingerprints and split identity checks.
- Supports probability, embedding, gradient-embedding, stochastic, committee, hybrid, and mix-style acquisition strategies.
- Integrates with Label Studio in external and managed Docker modes.
- Provides a deterministic simulator backend for tests and local smoke runs.
- Exports labels, dataset splits, reports, audit events, and benchmark evidence.
- Ships benchmark harnesses for synthetic and capped real-data validation.

### Current Status

This is a beta SDK. The core runtime, project state machine, strategy surface, simulator backend, Label Studio contracts, reports, exports, benchmark harness, and packaging checks are covered by tests.

Known limitations:

- The built-in Hugging Face adapter currently implements `predict_proba()` only. Training and evaluation must be supplied by your adapter subclass or custom adapter.
- LLM labeling backend support is a placeholder, not a production backend.
- Bandit scheduling is experimental.
- Benchmark evidence is useful validation evidence, not a universal scientific claim that every strategy beats random on every dataset.

### Installation

Core install:

```bash
pip install active-learning-sdk
```

Development install from this repository:

```bash
uv sync --dev
```

Optional extras:

```bash
pip install -e ".[sklearn]"
pip install -e ".[huggingface]"
pip install -e ".[datasets]"
pip install -e ".[xxhash]"
pip install -e ".[benchmarks]"
pip install -e ".[all]"
```

| Extra | Adds | Use case |
| --- | --- | --- |
| `sklearn` | `scikit-learn` | `SklearnTextClassifierAdapter` and local sklearn workflows. |
| `huggingface` | `transformers`, `torch`, `sentencepiece`, `protobuf` | Hugging Face probability adapter scaffolding. |
| `datasets` | `datasets`, `pandas` | Hugging Face datasets and DataFrame/CSV/Parquet workflows. |
| `xxhash` | `xxhash` | Faster dataset fingerprinting. |
| `benchmarks` | `datasets`, `pandas`, `scikit-learn` | Benchmark harnesses. |
| `all` | all optional runtime extras | Convenience install for local experimentation. |

## Core Simulator Quickstart

This quickstart does not require `pandas`, optional extras, or a live Label Studio service.

```python
from active_learning_sdk import (
    ActiveLearningProject,
    AnnotationPolicy,
    LabelBackendConfig,
    LabelSchema,
    SchedulerConfig,
    SplitConfig,
    StopCriteria,
)
from active_learning_sdk.backends.simulator import SimulatorLabelBackend
from active_learning_sdk.types import DataSample


class TinyProvider:
    def __init__(self):
        self.rows = {
            "s1": "free trial works",
            "s2": "invoice failed",
            "s3": "upgrade account",
            "s4": "refund request",
        }

    def iter_sample_ids(self):
        return iter(self.rows)

    def get_sample(self, sample_id):
        return DataSample(sample_id=sample_id, data={"text": self.rows[sample_id]})

    def schema(self):
        return {"sample_id": "str", "text": "str"}


class TinyModel:
    def __init__(self):
        self.labels = []

    def predict_proba(self, texts, batch_size=32):
        return [[0.5, 0.5] for _ in texts]

    def fit(self, texts, labels, **kwargs):
        self.labels = list(labels)

    def evaluate(self, texts, labels):
        return {"accuracy": 1.0 if labels else 0.0}

    def get_model_id(self):
        return f"tiny-model-{len(self.labels)}"


backend = SimulatorLabelBackend(
    label_by_sample_id={"s3": "positive", "s4": "negative"}
)

project = ActiveLearningProject("quickstart", workdir="./runs/quickstart")
project.configure(
    dataset=TinyProvider(),
    model=TinyModel(),
    label_schema=LabelSchema(
        task="text_classification",
        labels=["negative", "positive"],
    ),
    label_backend_config=LabelBackendConfig(backend="simulator"),
    label_backend=backend,
    scheduler_config=SchedulerConfig(mode="single", strategy="random"),
    annotation_policy=AnnotationPolicy(mode="latest", min_votes=1),
    split_config=SplitConfig(
        mode="explicit",
        explicit_splits={
            "train": ["s1", "s2", "s3", "s4"],
            "val": [],
            "test": [],
        },
    ),
)

project.import_labels({"s1": "positive", "s2": "negative"}, source="seed")
project.run(
    batch_size=2,
    stop_criteria=StopCriteria(max_rounds=1),
    poll_interval_seconds=0,
)
print(project.status()["counts"])
project.close()
```

Expected output:

```python
{"labeled": 4, "unlabeled": 0, "needs_review": 0, "invalid": 0}
```

### Core Concepts

An active learning project combines four pieces:

- Dataset provider: exposes stable `sample_id` values and returns `DataSample` objects.
- Model adapter: implements `fit()` and `evaluate()`; strategies that use probabilities also require `predict_proba()`.
- Label schema: defines task type and valid labels.
- Label backend: receives tasks and returns annotations.

The main facade is `ActiveLearningProject`. It persists state in `workdir`, so a project can be stopped and resumed. If you reopen a project in a new process, call `attach_runtime(...)` to bind the dataset, model, and optional backend instance back to the persisted state.

### Acquisition Strategies

Current strategy names include:

- `random`
- `entropy`
- `margin`
- `least_confidence`
- `group_diverse_entropy`
- `class_balanced_entropy`
- `class_group_balanced_entropy`
- `coreset_kcenter`
- `embedding_kmeans_pp`
- `max_min_embedding`
- `deduplicate_near_neighbors`
- `density_weighted_diversity`
- `badge`
- `adaptive_uncertainty_diversity`
- `mc_dropout_entropy`
- `bald`
- `variation_ratio`
- `prediction_variance`
- `committee_vote_entropy`
- `committee_kl_divergence`
- `committee_pairwise_disagreement`
- `committee_margin`

Scheduler modes:

- `single`: one strategy.
- `mix`: batch allocation across multiple strategy arms.
- `mix_interleaved`: interleaved multi-strategy allocation.
- `hybrid`: score blending, prefilters, rerankers, and guardrails.
- `custom`: injected user strategy.
- `bandit`: basic experimental adaptive arm selection.

### Label Studio

#### External Mode

External mode connects to a Label Studio service you run yourself:

```python
LabelBackendConfig(
    backend="label_studio",
    mode="external",
    url="http://127.0.0.1:8080",
    api_token="YOUR_LABEL_STUDIO_TOKEN",
)
```

Local Docker example:

```bash
docker run -d --name label-studio -p 8080:8080 heartexlabs/label-studio:1.23.0
```

#### Managed Docker Mode

Managed Docker mode uses packaged compose assets and requires explicit credentials:

```bash
export ACTIVE_LEARNING_LABEL_STUDIO_USERNAME="you@example.com"
export ACTIVE_LEARNING_LABEL_STUDIO_PASSWORD="change-me"
export ACTIVE_LEARNING_LABEL_STUDIO_TOKEN="change-this-token"
```

```python
LabelBackendConfig(
    backend="label_studio",
    mode="managed_docker",
    managed_port=8080,
    api_token="change-this-token",
)
```

See [docker/label_studio/README.md](docker/label_studio/README.md) and [docs/LABEL_STUDIO_LIVE_TESTS.md](docs/LABEL_STUDIO_LIVE_TESTS.md).

### Reports And Exports

The SDK can generate reproducible project artifacts:

- `summary.json`
- `report.md`
- `report.html`
- `manifest.json`
- label exports in JSONL or CSV
- dataset split exports in JSONL or CSV
- audit events and per-round selection snapshots

Typical public methods:

- `run(...)`
- `run_step(...)`
- `status()`
- `validate()`
- `list_rounds()`
- `get_round(round_id)`
- `import_labels(...)`
- `generate_report(...)`
- `export_labels(...)`
- `export_dataset_split(...)`
- `cache_stats()`
- `clear_cache(...)`

### Benchmarks

Benchmark entrypoints:

```bash
uv run python benchmarks/sdk_first_benchmark.py --preset smoke
uv run python benchmarks/sdk_first_benchmark.py --preset full
uv run python benchmarks/sdk_first_benchmark.py --preset project_smoke
uv run python benchmarks/sdk_first_benchmark.py --preset real_medium --seeds 13,21,34 --max-train-samples 800 --max-test-samples 500
```

Benchmark docs:

- [benchmarks/README.md](benchmarks/README.md)
- [docs/BENCHMARK_EVIDENCE.md](docs/BENCHMARK_EVIDENCE.md)
- [benchmarks/results/current_benchmark_report.md](benchmarks/results/current_benchmark_report.md)

Latest local validation in this worktree:

- `uv run pytest -q` -> `623 passed, 1 skipped`
- `uv run mypy src` -> `Success: no issues found in 38 source files`
- `uv run ruff check .` -> `All checks passed!`
- `uv build` -> wheel and source distribution built successfully
- `uv run --with twine python -m twine check dist/active_learning_sdk-0.1.0.tar.gz dist/active_learning_sdk-0.1.0-py3-none-any.whl` -> both artifacts passed

### Development

```bash
uv sync --dev
uv run pytest -q
uv run ruff check .
uv run mypy src
uv build
```

Community files:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [SECURITY.md](SECURITY.md)
- [SUPPORT.md](SUPPORT.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [CHANGELOG.md](CHANGELOG.md)

### License

Apache License 2.0. See [LICENSE](LICENSE).

---

<a id="russian"></a>

## Русский

`active-learning-sdk` — Python SDK для построения возобновляемых циклов активного обучения в задачах текстовой классификации. SDK хранит состояние проекта на диске, выбирает следующие примеры для разметки, отправляет их в backend разметки, забирает аннотации, переобучает ваш model adapter и сохраняет воспроизводимые отчеты.

Проект нужен командам, которые хотят использовать active learning как надежный компонент рабочего процесса, а не как набор хрупких notebook-скриптов.

### Что Умеет SDK

- Запускает stateful active learning раунды: select, push, wait, pull, train/evaluate, update.
- Возобновляет прерванные проекты без повторного создания задач разметки.
- Защищает проект через fingerprint датасета и проверку идентичности split-ов.
- Поддерживает probability, embedding, gradient-embedding, stochastic, committee, hybrid и mix стратегии выбора примеров.
- Интегрируется с Label Studio в external и managed Docker режимах.
- Содержит deterministic simulator backend для тестов и локальных smoke-прогонов.
- Экспортирует labels, dataset splits, reports, audit events и benchmark evidence.
- Содержит benchmark harness для synthetic и capped real-data проверок.

### Текущий Статус

Это beta SDK. Основной runtime, state machine проекта, стратегии, simulator backend, Label Studio контракты, отчеты, exports, benchmark harness и упаковка покрыты тестами.

Ограничения:

- Встроенный Hugging Face adapter сейчас реализует только `predict_proba()`. `fit()` и `evaluate()` должны быть реализованы в вашем subclass/custom adapter.
- LLM backend пока placeholder, а не production backend.
- Bandit scheduler экспериментальный.
- Benchmark evidence показывает полезную проверку качества, но не является универсальным научным утверждением, что каждая стратегия всегда лучше random на любом датасете.

### Установка

Базовая установка:

```bash
pip install active-learning-sdk
```

Локальная разработка из репозитория:

```bash
uv sync --dev
```

Опциональные зависимости:

```bash
pip install -e ".[sklearn]"
pip install -e ".[huggingface]"
pip install -e ".[datasets]"
pip install -e ".[xxhash]"
pip install -e ".[benchmarks]"
pip install -e ".[all]"
```

| Extra | Что добавляет | Для чего |
| --- | --- | --- |
| `sklearn` | `scikit-learn` | `SklearnTextClassifierAdapter` и локальные sklearn workflows. |
| `huggingface` | `transformers`, `torch`, `sentencepiece`, `protobuf` | Hugging Face probability adapter scaffold. |
| `datasets` | `datasets`, `pandas` | Hugging Face datasets и DataFrame/CSV/Parquet workflows. |
| `xxhash` | `xxhash` | Более быстрый fingerprint датасета. |
| `benchmarks` | `datasets`, `pandas`, `scikit-learn` | Benchmark harness. |
| `all` | все optional runtime extras | Удобно для локальных экспериментов. |

### Быстрый Старт С Simulator Backend

Этот пример не требует pandas, scikit-learn, Hugging Face, Docker или Label Studio.

```python
from active_learning_sdk import (
    ActiveLearningProject,
    AnnotationPolicy,
    LabelBackendConfig,
    LabelSchema,
    SchedulerConfig,
    SplitConfig,
    StopCriteria,
)
from active_learning_sdk.backends.simulator import SimulatorLabelBackend
from active_learning_sdk.types import DataSample


class TinyProvider:
    def __init__(self):
        self.rows = {
            "s1": "free trial works",
            "s2": "invoice failed",
            "s3": "upgrade account",
            "s4": "refund request",
        }

    def iter_sample_ids(self):
        return iter(self.rows)

    def get_sample(self, sample_id):
        return DataSample(sample_id=sample_id, data={"text": self.rows[sample_id]})

    def schema(self):
        return {"sample_id": "str", "text": "str"}


class TinyModel:
    def __init__(self):
        self.labels = []

    def predict_proba(self, texts, batch_size=32):
        return [[0.5, 0.5] for _ in texts]

    def fit(self, texts, labels, **kwargs):
        self.labels = list(labels)

    def evaluate(self, texts, labels):
        return {"accuracy": 1.0 if labels else 0.0}

    def get_model_id(self):
        return f"tiny-model-{len(self.labels)}"


backend = SimulatorLabelBackend(
    label_by_sample_id={"s3": "positive", "s4": "negative"}
)

project = ActiveLearningProject("quickstart", workdir="./runs/quickstart")
project.configure(
    dataset=TinyProvider(),
    model=TinyModel(),
    label_schema=LabelSchema(
        task="text_classification",
        labels=["negative", "positive"],
    ),
    label_backend_config=LabelBackendConfig(backend="simulator"),
    label_backend=backend,
    scheduler_config=SchedulerConfig(mode="single", strategy="random"),
    annotation_policy=AnnotationPolicy(mode="latest", min_votes=1),
    split_config=SplitConfig(
        mode="explicit",
        explicit_splits={
            "train": ["s1", "s2", "s3", "s4"],
            "val": [],
            "test": [],
        },
    ),
)

project.import_labels({"s1": "positive", "s2": "negative"}, source="seed")
project.run(
    batch_size=2,
    stop_criteria=StopCriteria(max_rounds=1),
    poll_interval_seconds=0,
)
print(project.status()["counts"])
project.close()
```

Ожидаемый результат:

```python
{"labeled": 4, "unlabeled": 0, "needs_review": 0, "invalid": 0}
```

### Основные Понятия

Active learning проект состоит из четырех частей:

- Dataset provider: выдает стабильные `sample_id` и возвращает `DataSample`.
- Model adapter: реализует `fit()` и `evaluate()`; стратегии на вероятностях требуют `predict_proba()`.
- Label schema: описывает тип задачи и допустимые labels.
- Label backend: получает задачи и возвращает аннотации.

Главная публичная точка входа — `ActiveLearningProject`. Она сохраняет состояние в `workdir`, поэтому проект можно остановить и продолжить. Если вы открываете проект в новом процессе, используйте `attach_runtime(...)`, чтобы снова привязать dataset, model и backend к сохраненному состоянию.

### Стратегии Выбора

Доступные strategy names:

- `random`
- `entropy`
- `margin`
- `least_confidence`
- `group_diverse_entropy`
- `class_balanced_entropy`
- `class_group_balanced_entropy`
- `coreset_kcenter`
- `embedding_kmeans_pp`
- `max_min_embedding`
- `deduplicate_near_neighbors`
- `density_weighted_diversity`
- `badge`
- `adaptive_uncertainty_diversity`
- `mc_dropout_entropy`
- `bald`
- `variation_ratio`
- `prediction_variance`
- `committee_vote_entropy`
- `committee_kl_divergence`
- `committee_pairwise_disagreement`
- `committee_margin`

Режимы scheduler:

- `single`: одна стратегия.
- `mix`: batch allocation между несколькими strategy arms.
- `mix_interleaved`: interleaved multi-strategy allocation.
- `hybrid`: score blending, prefilters, rerankers и guardrails.
- `custom`: пользовательская стратегия.
- `bandit`: базовый экспериментальный adaptive arm selection.

### Label Studio

External Label Studio:

```python
LabelBackendConfig(
    backend="label_studio",
    mode="external",
    url="http://127.0.0.1:8080",
    api_token="YOUR_LABEL_STUDIO_TOKEN",
)
```

Локальный Docker:

```bash
docker run -d --name label-studio -p 8080:8080 heartexlabs/label-studio:1.23.0
```

Managed Docker mode использует packaged compose assets и явные credentials:

```bash
export ACTIVE_LEARNING_LABEL_STUDIO_USERNAME="you@example.com"
export ACTIVE_LEARNING_LABEL_STUDIO_PASSWORD="change-me"
export ACTIVE_LEARNING_LABEL_STUDIO_TOKEN="change-this-token"
```

```python
LabelBackendConfig(
    backend="label_studio",
    mode="managed_docker",
    managed_port=8080,
    api_token="change-this-token",
)
```

Подробнее:

- [docker/label_studio/README.md](docker/label_studio/README.md)
- [docs/LABEL_STUDIO_LIVE_TESTS.md](docs/LABEL_STUDIO_LIVE_TESTS.md)

### Reports И Exports

SDK умеет генерировать воспроизводимые artifacts:

- `summary.json`
- `report.md`
- `report.html`
- `manifest.json`
- label exports в JSONL или CSV
- dataset split exports в JSONL или CSV
- audit events и per-round selection snapshots

Полезные публичные методы:

- `run(...)`
- `run_step(...)`
- `status()`
- `validate()`
- `list_rounds()`
- `get_round(round_id)`
- `import_labels(...)`
- `generate_report(...)`
- `export_labels(...)`
- `export_dataset_split(...)`
- `cache_stats()`
- `clear_cache(...)`

### Benchmarks

Основные команды:

```bash
uv run python benchmarks/sdk_first_benchmark.py --preset smoke
uv run python benchmarks/sdk_first_benchmark.py --preset full
uv run python benchmarks/sdk_first_benchmark.py --preset project_smoke
uv run python benchmarks/sdk_first_benchmark.py --preset real_medium --seeds 13,21,34 --max-train-samples 800 --max-test-samples 500
```

Документация:

- [benchmarks/README.md](benchmarks/README.md)
- [docs/BENCHMARK_EVIDENCE.md](docs/BENCHMARK_EVIDENCE.md)
- [benchmarks/results/current_benchmark_report.md](benchmarks/results/current_benchmark_report.md)

Последняя локальная проверка в этом worktree:

- `uv run pytest -q` -> `623 passed, 1 skipped`
- `uv run mypy src` -> `Success: no issues found in 38 source files`
- `uv run ruff check .` -> `All checks passed!`
- `uv build` -> wheel и source distribution собраны успешно
- `uv run --with twine python -m twine check dist/active_learning_sdk-0.1.0.tar.gz dist/active_learning_sdk-0.1.0-py3-none-any.whl` -> оба artifact прошли проверку

### Разработка

```bash
uv sync --dev
uv run pytest -q
uv run ruff check .
uv run mypy src
uv build
```

Файлы сообщества:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [SECURITY.md](SECURITY.md)
- [SUPPORT.md](SUPPORT.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [CHANGELOG.md](CHANGELOG.md)

### Лицензия

Apache License 2.0. См. [LICENSE](LICENSE).
