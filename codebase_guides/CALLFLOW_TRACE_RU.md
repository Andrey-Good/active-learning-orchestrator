# Трасса вызовов и поток данных (RU)

Этот файл — **не объяснение логики** (почему так сделано), а именно “трасса”:

- какие функции вызываются,
- какие аргументы они принимают,
- что они возвращают (или что возвращают вызываемые ими функции),
- какие методы вызывают другие методы.

Цель: увидеть “как ток течёт по проводам” от пользовательского кода внутрь SDK.

---

## 0) Стартовый пользовательский код (внешний мир)

Обычно пользователь пишет примерно так:

```python
from active_learning_sdk import ActiveLearningProject
from active_learning_sdk import LabelSchema, LabelBackendConfig, SchedulerConfig

project = ActiveLearningProject(project_name, workdir=...)

project.configure(
    dataset=...,                 # DatasetProvider | pandas.DataFrame | str/Path
    model=...,                   # TextClassificationAdapter
    label_schema=LabelSchema(...),
    label_backend_config=LabelBackendConfig(...),
    scheduler_config=SchedulerConfig(...),
)

project.run(batch_size=..., stop_criteria=..., poll_interval_seconds=...)
```

Дальше — трасса того, что вызывается внутри.

---

## 1) `ActiveLearningProject(...)` (фасад, публичная точка входа)

Файл: `src/active_learning_sdk/project.py`

### 1.1 `ActiveLearningProject.__init__`

Вызов пользователя:

```
ActiveLearningProject.__init__(
    project_name: str,
    workdir: Union[str, Path],
    *,
    state_store: Optional[StateStore] = None,
    lock: bool = True,
) -> None
```

Внутренние вызовы:

1) `ActiveLearningEngine.__init__(project_name, workdir, state_store=..., lock=...) -> None`

Возврат наружу:
- `None` (конструктор)
- но пользователь получает объект `ActiveLearningProject` с полем:
  - `self._engine: ActiveLearningEngine`

---

## 2) `project.configure(...)` (приём входных объектов и запись конфигов в state)

Файлы: `src/active_learning_sdk/project.py`, `src/active_learning_sdk/engine.py`

### 2.1 `ActiveLearningProject.configure`

Вызов пользователя:

```
ActiveLearningProject.configure(
    *,
    dataset: Union[DatasetProvider, Any, str, Path],
    model: TextClassificationAdapter,
    label_schema: LabelSchema,
    label_backend_config: LabelBackendConfig,
    scheduler_config: SchedulerConfig,
    label_backend: Optional[LabelBackend] = None,
    annotation_policy: AnnotationPolicy = AnnotationPolicy(),
    cache_config: CacheConfig = CacheConfig(),
    fingerprint_config: FingerprintConfig = FingerprintConfig(),
    split_config: SplitConfig = SplitConfig(),
    prelabel_config: PrelabelConfig = PrelabelConfig(),
) -> None
```

Внутренние вызовы (фасад → engine):

1) `ActiveLearningEngine.configure(...) -> None`

Возврат наружу:
- `None`

### 2.2 `ActiveLearningEngine.configure`

Файл: `src/active_learning_sdk/engine.py`

Сигнатура:

```
ActiveLearningEngine.configure(
    *,
    dataset: Union[DatasetProvider, Any, str, Path],
    model: TextClassificationAdapter,
    label_schema: LabelSchema,
    label_backend_config: LabelBackendConfig,
    scheduler_config: SchedulerConfig,
    label_backend: Optional[LabelBackend] = None,
    annotation_policy: AnnotationPolicy = AnnotationPolicy(),
    cache_config: CacheConfig = CacheConfig(),
    fingerprint_config: FingerprintConfig = FingerprintConfig(),
    split_config: SplitConfig = SplitConfig(),
    prelabel_config: PrelabelConfig = PrelabelConfig(),
) -> None
```

Внутренние вызовы (по порядку, упрощённо):

1) `_ensure_state_loaded() -> None`
   - если нужно, вызывает:
     - `StateStore.load() -> ProjectState`
       - в дефолте: `JsonFileStateStore.load() -> ProjectState`
         - использует `state_from_json_dict(payload: Dict[str, Any]) -> ProjectState`

2) Валидации конфигов (все возвращают `None` или кидают исключение):
   - `label_schema.validate() -> None`
   - `annotation_policy.validate() -> None`
   - `scheduler_config.validate() -> None`
   - `fingerprint_config.validate() -> None`
   - `split_config.validate() -> None`
   - `prelabel_config.validate() -> None`
   - `label_backend_config.validate() -> None`

3) Нормализация датасета:
   - `provider = _coerce_dataset(dataset) -> DatasetProvider`
     - возможные ветки:
       - если `dataset` уже `DatasetProvider`: возвращает его
       - если `dataset` похож на DataFrame: возвращает `DataFrameDatasetProvider(dataset)`
       - если `dataset` это `str|Path`: читает CSV/Parquet и возвращает `DataFrameDatasetProvider.from_path(path)`

4) Fingerprint датасета:
   - `fingerprinter = DatasetFingerprinter(fingerprint_config) -> DatasetFingerprinter`
   - `fp = fingerprinter.fingerprint(provider) -> str`

5) Проверка/инспекция модели:
   - `caps = inspect_model_capabilities(model) -> ModelCapabilities`
   - `_validate_model_capabilities(caps: ModelCapabilities) -> None`

6) Кэши:
   - `_init_caches(cache_config: CacheConfig) -> None`
     - создаёт `PredictionCache(...)` и `EmbeddingCache(...)` (или отключает)

7) Backend:
   - если `label_backend` передан: `backend = label_backend`
   - иначе:
     - `backend = build_label_backend(label_backend_config) -> LabelBackend`
   - `backend.ensure_ready(label_schema) -> Dict[str, Any]`

8) Aggregator:
   - `self._aggregator = AnnotationAggregator(annotation_policy) -> AnnotationAggregator`

9) Scheduler:
   - `scheduler = StrategyScheduler(scheduler_config, strategies=[...]) -> StrategyScheduler`
   - `self._scheduler = scheduler`

10) Формирование и запись `ProjectState` (все эти операции сами по себе ничего не “возвращают”, это присваивания):
   - `self._state.dataset_ref = DatasetRef(...)`
     - вызывает `provider.schema() -> Dict[str, str]`
   - `self._state.label_schema = dataclasses.asdict(label_schema) -> Dict[str, Any]`
   - `self._state.annotation_policy = dataclasses.asdict(annotation_policy) -> Dict[str, Any]`
   - `self._state.scheduler_config = dataclasses.asdict(scheduler_config) -> Dict[str, Any]`
   - `self._state.label_backend_config = dataclasses.asdict(label_backend_config) -> Dict[str, Any]`
   - `self._state.cache_config = dataclasses.asdict(cache_config) -> Dict[str, Any]`
   - `self._state.split_config = dataclasses.asdict(split_config) -> Dict[str, Any]`
   - `self._state.prelabel_config = dataclasses.asdict(prelabel_config) -> Dict[str, Any]`

11) Splits:
   - `self._state.splits = _resolve_splits(provider, split_config) -> Dict[str, List[str]]`
     - использует `provider.iter_sample_ids() -> Iterator[str]`

12) Инициализация статусов (если первый конфиг):
   - `provider.iter_sample_ids() -> Iterator[str]`
   - заполняет:
     - `self._state.sample_status: Dict[str, str]`

13) Сохранение state:
   - `_touch_state() -> None`
   - `_save_state() -> None`
     - `StateStore.save_atomic(state: ProjectState) -> None`
       - в дефолте: `JsonFileStateStore.save_atomic(state) -> None`
         - `state_to_json_dict(state: ProjectState) -> Dict[str, Any]`
           - `dataclass_to_dict(obj: Any) -> Any`
         - `atomic_write_text(path, payload) -> None`

Возврат наружу:
- `None`

---

## 3) `project.run(...)` (блокирующий цикл поверх `run_step`)

Файлы: `src/active_learning_sdk/project.py`, `src/active_learning_sdk/engine.py`

### 3.1 `ActiveLearningProject.run`

Сигнатура:

```
ActiveLearningProject.run(
    *,
    budget: Optional[int] = None,
    batch_size: int = 50,
    stop_criteria: StopCriteria = StopCriteria(),
    resume: bool = True,
    poll_interval_seconds: int = 10,
) -> None
```

Внутренний вызов:
- `ActiveLearningEngine.run(...) -> None`

Возврат наружу:
- `None`

### 3.2 `ActiveLearningEngine.run`

Сигнатура:

```
ActiveLearningEngine.run(
    *,
    budget: Optional[int] = None,
    batch_size: int = 50,
    stop_criteria: StopCriteria = StopCriteria(),
    resume: bool = True,
    poll_interval_seconds: int = 10,
) -> None
```

Внутренние вызовы (цикл):

1) `_ensure_configured() -> None`
   - внутри вызывает `_ensure_state_loaded() -> None` (если надо)
   - проверяет, что runtime-объекты (`_provider/_model/_label_backend/...`) не `None`

2) Если `budget is not None`:
   - `stop_criteria = dataclasses.replace(stop_criteria, max_labeled=budget) -> StopCriteria`

3) `stop_criteria.validate() -> None`

4) `while True:` (бесконечный цикл до stop)
   - `if _should_stop(stop_criteria) -> bool`: если True → `break`
   - иначе:
     - `result = run_step(batch_size=..., poll_interval_seconds=...) -> StepResult`
     - если `result.step == StepKind.NOOP`:
       - `time.sleep(poll_interval_seconds) -> None`
       - `continue`

Возврат наружу:
- `None`

---

## 4) `project.run_step(...)` (один шаг state-machine, основной “провод”)

Файлы: `src/active_learning_sdk/project.py`, `src/active_learning_sdk/engine.py`

### 4.1 `ActiveLearningProject.run_step`

Сигнатура:

```
ActiveLearningProject.run_step(
    *,
    batch_size: int = 50,
    poll_interval_seconds: int = 0,
) -> StepResult
```

Внутренний вызов:
- `ActiveLearningEngine.run_step(batch_size=..., poll_interval_seconds=...) -> StepResult`

Возврат наружу:
- `StepResult(step: StepKind, round_id: Optional[str], message: str, details: Dict[str, Any])`

### 4.2 `ActiveLearningEngine.run_step`

Файл: `src/active_learning_sdk/engine.py`

Сигнатура:

```
ActiveLearningEngine.run_step(
    *,
    batch_size: int = 50,
    poll_interval_seconds: int = 0,
) -> StepResult
```

Внутренние вызовы (по коду буквально):

1) `_ensure_configured() -> None`

2) `round_state = _get_or_create_active_round() -> RoundState`
   - если активный раунд уже есть → возвращает его
   - иначе создаёт новый `RoundState(...)` и сохраняет state:
     - `_touch_state() -> None`
     - `_save_state() -> None`

3) `next_step = _next_step(round_state: RoundState) -> StepKind`

4) Ветка по `next_step`:

#### 4.2.A) SELECT

Вызов:
- `_step_select(round_state: RoundState, *, batch_size: int) -> None`

Возврат наружу:
- `StepResult(step=StepKind.SELECT, round_id=str, message=str, details={})`

#### 4.2.B) PUSH

Вызов:
- `_step_push(round_state: RoundState) -> None`

Возврат наружу:
- `StepResult(step=StepKind.PUSH, round_id=str, message=str, details={})`

#### 4.2.C) WAIT

Вызов:
- `progress = _step_wait(round_state: RoundState) -> RoundProgress`
- если `poll_interval_seconds > 0`:
  - `time.sleep(poll_interval_seconds) -> None`

Возврат наружу:
- `StepResult(step=StepKind.WAIT, round_id=str, message=str, details=dataclass_to_dict(progress))`
  - `dataclass_to_dict(progress: RoundProgress) -> Dict[str, Any]`

#### 4.2.D) PULL

Вызов:
- `_step_pull(round_state: RoundState) -> None`

Возврат наружу:
- `StepResult(step=StepKind.PULL, round_id=str, message=str, details={})`

#### 4.2.E) TRAIN_EVAL

Вызов:
- `_step_train_eval(round_state: RoundState) -> None`

Возврат наружу:
- `StepResult(step=StepKind.TRAIN_EVAL, round_id=str, message=str, details={})`

#### 4.2.F) UPDATE

Вызов:
- `_step_update(round_state: RoundState) -> None`

Возврат наружу:
- `StepResult(step=StepKind.UPDATE, round_id=str, message=str, details={})`

#### 4.2.G) NOOP

Возврат наружу:
- `StepResult(step=StepKind.NOOP, round_id=Optional[str], message=str, details={})`

---

## 5) Внутренние шаги (что они вызывают и что возвращают)

Ниже — те же шаги, но “внутри”: какие методы вызывают и что получают/отдают.

### 5.1 `_step_select(round_state, batch_size) -> None`

Файл: `src/active_learning_sdk/engine.py`

Внутренние вызовы:

1) Выбор pool:
   - читает `self._state.sample_status: Dict[str, str]` (не вызов функции)

2) Формирование `SelectionContext.__init__(...) -> None`:

```
SelectionContext(
    provider: DatasetProvider,
    model: TextClassificationAdapter,
    label_schema: LabelSchema,
    prediction_cache: Optional[PredictionCache],
    embedding_cache: Optional[EmbeddingCache],
    labeled_ids: Sequence[str],
    last_metrics: Dict[str, float],
) -> None
```

3) Выбор батча:

```
selected_ids, snapshot = StrategyScheduler.select_batch(
    pool_ids: Sequence[str],
    k: int,
    context: SelectionContext,
    state: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]
```

4) Внутри `StrategyScheduler.select_batch(...)` (упрощённо, по контракту):
- может вызвать:
  - `SamplingStrategy.select(pool_ids: Sequence[str], k: int, context: SelectionContext) -> List[str]`

5) У стратегий uncertainty внутри вызывается `context.predict_proba(sample_ids) -> Any`:

```
SelectionContext.predict_proba(
    sample_ids: Sequence[str],
    batch_size: int = 32,
) -> Any
```

А `predict_proba` в итоге вызывает адаптер:

```
TextClassificationAdapter.predict_proba(
    texts: Sequence[str],
    batch_size: int = 32,
) -> Any
```

И для получения текстов вызывается:
- `SelectionContext.get_samples(sample_ids) -> List[DataSample]`
  - `DatasetProvider.get_samples(sample_ids) -> List[DataSample]`
    - по дефолту: это список вызовов `DatasetProvider.get_sample(sample_id) -> DataSample`

Возвраты:
- `_step_select(...)` возвращает `None`, но записывает в `RoundState`:
  - `round_state.selected_sample_ids: List[str]`
  - `round_state.scheduler_snapshot: Dict[str, Any]`

### 5.2 `_step_push(round_state) -> None`

Внутренние вызовы:

1) Получить samples:
   - `samples = DatasetProvider.get_samples(sample_ids: Sequence[str]) -> List[DataSample]`

2) (опционально) prelabels:
   - `_make_prelabels(samples: Sequence[DataSample]) -> Dict[str, Any]`
     - вызывает:
       - `TextClassificationAdapter.predict_proba(texts: Sequence[str]) -> Any`

3) Backend push:

```
RoundPushResult = LabelBackend.push_round(
    round_id: str,
    samples: Sequence[DataSample],
    prelabels: Optional[Dict[str, Any]] = None,
) -> RoundPushResult
```

Возвраты:
- `_step_push(...) -> None`
- но сохраняет:
  - `round_state.task_ids: Dict[str, str]` (из `RoundPushResult.task_ids`)

### 5.3 `_step_wait(round_state) -> RoundProgress`

Внутренний вызов:

```
progress = LabelBackend.poll_round(
    round_id: str,
    task_ids: Mapping[str, str],
    policy: AnnotationPolicy,
) -> RoundProgress
```

Возврат:
- `RoundProgress(total: int, done: int, ready_sample_ids: List[str], details: Dict[str, Any])`

### 5.4 `_step_pull(round_state) -> None`

Внутренние вызовы:

1) Backend pull:

```
pull = LabelBackend.pull_round(
    round_id: str,
    task_ids: Mapping[str, str],
) -> RoundPullResult
```

2) Aggregation per sample:

```
resolved = AnnotationAggregator.resolve(
    sample_id: str,
    annotations: Sequence[AnnotationRecord],
) -> ResolvedLabel
```

Возвраты:
- `_step_pull(...) -> None`
- но обновляет state:
  - `ProjectState.sample_status[sample_id] = SampleStatus.value`
  - `ProjectState.sample_labels[sample_id] = Any`
  - `round_state.resolved: Dict[str, Any]`

### 5.5 `_step_train_eval(round_state) -> None`

Внутренние вызовы:

1) Получение train/val samples:
- `DatasetProvider.get_samples(sample_ids) -> List[DataSample]`

2) Обучение:

```
TextClassificationAdapter.fit(
    texts: Sequence[str],
    labels: Sequence[Any],
    **kwargs,
) -> None
```

3) Оценка:

```
TextClassificationAdapter.evaluate(
    texts: Sequence[str],
    labels: Sequence[Any],
) -> Dict[str, float]
```

Возвраты:
- `_step_train_eval(...) -> None`
- но:
  - сохраняет метрики в `ProjectState.metrics_history: List[MetricRecord]`

### 5.6 `_step_update(round_state) -> None`

Внутренние вызовы:

1) `reward = _compute_reward(round_state: RoundState) -> float`

2) Обновление состояния шедулера:

```
new_state = StrategyScheduler.update_reward(
    reward: float,
    snapshot: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]
```

Возврат:
- `_step_update(...) -> None`
- но обновляет:
  - `ProjectState.scheduler_state: Dict[str, Any]`

---

## 6) Вспомогательные цепочки (State I/O и Dataset I/O)

### 6.1 State load/save (JSON)

Файл: `src/active_learning_sdk/state/store.py`

- `JsonFileStateStore.load() -> ProjectState`
  - `json.loads(text) -> Dict[str, Any]`
  - `state_from_json_dict(payload: Dict[str, Any]) -> ProjectState`

- `JsonFileStateStore.save_atomic(state: ProjectState) -> None`
  - `state_to_json_dict(state: ProjectState) -> Dict[str, Any]`
    - `dataclass_to_dict(obj: Any) -> Any` (файл `src/active_learning_sdk/utils.py`)
  - `json.dumps(payload) -> str`
  - `atomic_write_text(path, payload) -> None` (файл `src/active_learning_sdk/utils.py`)

### 6.2 Dataset (DataFrame)

Файл: `src/active_learning_sdk/dataset/provider.py`

- `DataFrameDatasetProvider.iter_sample_ids() -> Iterator[str]`
- `DataFrameDatasetProvider.get_sample(sample_id: str) -> DataSample`
- `DataFrameDatasetProvider.get_samples(sample_ids: Sequence[str]) -> List[DataSample]` (унаследованное поведение через Protocol-default)

---

## 7) Что пользователь “получает на руки” после `run(...)`

`run(...)` сам по себе возвращает `None`, но пользователь может вызвать:

- `project.get_state() -> ProjectState`
  - внутри: `ActiveLearningEngine.get_state() -> ProjectState`

- `project.export_labels(output_path, format="jsonl") -> None`
  - внутри: `ActiveLearningEngine.export_labels(...) -> None`
  - пишет файл (side-effect), возвращает `None`

- `project.export_dataset_split(output_dir, which="labeled", format="jsonl") -> None`
  - внутри: `ActiveLearningEngine.export_dataset_split(...) -> None`

---

## 8) Короткая “сводка” трассы одним списком (без деталей)

1) `ActiveLearningProject.__init__ -> ActiveLearningEngine.__init__`
2) `ActiveLearningProject.configure -> ActiveLearningEngine.configure`
3) `ActiveLearningProject.run -> ActiveLearningEngine.run`
4) `ActiveLearningEngine.run` (loop) → `ActiveLearningEngine.run_step`
5) `run_step`:
   - `_get_or_create_active_round`
   - `_next_step`
   - `_step_select` OR `_step_push` OR `_step_wait` OR `_step_pull` OR `_step_train_eval` OR `_step_update`
   - возвращает `StepResult`

