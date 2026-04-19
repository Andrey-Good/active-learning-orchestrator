# Walkthrough: как устроен Active Learning SDK (для джуна)

Этот файл объясняет, как работает код в репозитории, чтобы ты:

- не терялся в папках и файлах,
- понимал, что уже реализовано, а что пока заглушка,
- видел цепочки вызовов: от методов, которыми пользуется человек, до внутренних функций,
- не писал дублирующую логику поверх уже существующей.

В тексте я даю ссылки вида `путь/файл.py:строка` (1-based). Открыв файл и прыгнув на строку, ты увидишь точное место.

## 1) С чего начать смотреть код

Главные 2 точки входа:

- Публичный класс для пользователя: `src/active_learning_sdk/project.py:25` (`ActiveLearningProject`).
- Внутренний движок со всей логикой цикла: `src/active_learning_sdk/engine.py:376` (`ActiveLearningEngine`).

Важно: `ActiveLearningProject` почти ничего не делает сам. Он делегирует в `ActiveLearningEngine`.

Если ты хочешь понять "что вызывается при `project.run()`", тебе нужно читать:

1. `ActiveLearningProject.run` -> делегирует в `ActiveLearningEngine.run`.
2. `ActiveLearningEngine.run` -> много раз вызывает `ActiveLearningEngine.run_step`.
3. `ActiveLearningEngine.run_step` -> выбирает следующий шаг state machine и вызывает приватные `_step_*`.

## 2) Карта репозитория (что где лежит)

Ядро пакета в `src/active_learning_sdk/`:

- `src/active_learning_sdk/__init__.py` — что импортирует пользователь "сверху" (публичный фасад).
- `src/active_learning_sdk/project.py` — тонкий фасад `ActiveLearningProject` (делегирует в engine).
- `src/active_learning_sdk/engine.py` — реальная логика оркестратора и state machine.
- `src/active_learning_sdk/configs.py` — dataclass-конфиги, которые передаются в `configure()`.
- `src/active_learning_sdk/types.py` — enum'ы и базовые структуры (samples, annotations, metrics).
- `src/active_learning_sdk/exceptions.py` — все исключения SDK (что может вылетать наружу).
- `src/active_learning_sdk/utils.py` — мелкие утилиты (например atomic write).

Подсистемы:

- State и идемпотентность:
  - `src/active_learning_sdk/state/lock.py` — файловый lock на проект.
  - `src/active_learning_sdk/state/store.py` — схема state и JSON store (load/save).
- Dataset:
  - `src/active_learning_sdk/dataset/provider.py` — доступ к данным (`DatasetProvider`, `DataFrameDatasetProvider`).
  - `src/active_learning_sdk/dataset/fingerprint.py` — fingerprint датасета.
- Стратегии и выбор:
  - `src/active_learning_sdk/strategies/base.py` — протокол `SamplingStrategy`.
  - `src/active_learning_sdk/strategies/uncertainty.py` — Random/Entropy/Margin/LeastConfidence + placeholder diversity.
- Backends разметки:
  - `src/active_learning_sdk/backends/base.py` — протокол `LabelBackend`, типы результатов, `build_label_backend`.
  - `src/active_learning_sdk/backends/label_studio.py` — scaffold Label Studio (пока заглушки).
- Модельные адаптеры:
  - `src/active_learning_sdk/adapters/base.py` — протокол `TextClassificationAdapter`, `inspect_model_capabilities`.
  - `src/active_learning_sdk/adapters/huggingface.py` — scaffold `HFSequenceClassifierAdapter`.
- Кеш:
  - `src/active_learning_sdk/cache.py` — stores + PredictionCache + EmbeddingCache.
- Агрегация аннотаций:
  - `src/active_learning_sdk/annotation.py` — `AnnotationAggregator`.
- Отчеты:
  - `src/active_learning_sdk/report.py` — `ReportGenerator` (пока заглушка).

## 3) Публичные методы: что видит пользователь

Пользователь обычно делает 4 вещи:

1. Создает проект: `ActiveLearningProject(...)`
2. Конфигурирует: `project.configure(...)`
3. Запускает цикл: `project.run(...)` или по шагам `project.run_step(...)`
4. Смотрит состояние/экспортирует/строит отчет: `status()/get_state()/export_*/generate_report()`

Все эти методы находятся тут: `src/active_learning_sdk/project.py:25`.

### 3.1) ActiveLearningProject.__init__

`ActiveLearningProject.__init__` (`src/active_learning_sdk/project.py:28`) делает одно:

- создает `ActiveLearningEngine` и кладет его в `self._engine`.

То есть фактически "проект" — это обертка над движком.

### 3.2) ActiveLearningProject.configure

`ActiveLearningProject.configure` (`src/active_learning_sdk/project.py:52`) просто прокидывает параметры в:

- `ActiveLearningEngine.configure` (`src/active_learning_sdk/engine.py:483`).

Поэтому настоящую логику "что происходит при configure" смотри в engine.

### 3.3) ActiveLearningProject.run / run_step

- `ActiveLearningProject.run` (`src/active_learning_sdk/project.py:93`) -> `ActiveLearningEngine.run` (`src/active_learning_sdk/engine.py:730`)
- `ActiveLearningProject.run_step` (`src/active_learning_sdk/project.py:110`) -> `ActiveLearningEngine.run_step` (`src/active_learning_sdk/engine.py:786`)

### 3.4) Методы "сервисные"

Все это тоже делегирование:

- `status` (`src/active_learning_sdk/project.py:113`) -> `engine.status` (`src/active_learning_sdk/engine.py:852`)
- `get_state` (`src/active_learning_sdk/project.py:116`) -> `engine.get_state` (`src/active_learning_sdk/engine.py:880`)
- `export_labels` (`src/active_learning_sdk/project.py:131`) -> `engine.export_labels` (`src/active_learning_sdk/engine.py:960`)
- `generate_report` (`src/active_learning_sdk/project.py:128`) -> `engine.generate_report` (`src/active_learning_sdk/engine.py:943`)

## 4) Внутренний движок: ActiveLearningEngine

Все важное происходит в `src/active_learning_sdk/engine.py:376`.

У движка есть:

- Персистентный state (`ProjectState`) в `state.json` внутри `workdir`.
- Lock-файл `project.lock` для защиты от параллельных запусков.
- Живые runtime-объекты: dataset provider, model adapter, backend, scheduler, caches и т.д.
- State machine раунда: SELECT -> PUSH -> WAIT -> PULL -> TRAIN_EVAL -> UPDATE.

### 4.1) Engine.__init__: открытие проекта

`ActiveLearningEngine.__init__` (`src/active_learning_sdk/engine.py:412`) делает:

1. Создает `workdir` (если нет).
2. Если lock включен, захватывает lock:
   - `ProjectLock.acquire` (`src/active_learning_sdk/state/lock.py:19`)
3. Определяет путь до state: `workdir/state.json`.
4. Если state уже есть, загружает через store:
   - `JsonFileStateStore.load` (`src/active_learning_sdk/state/store.py:152`)
   - `state_from_json_dict` (`src/active_learning_sdk/state/store.py:106`)
   - потом проверяет базовую валидность:
     - `_validate_loaded_state_basic` (`src/active_learning_sdk/engine.py:1141`)
5. Если state нет, создает новый:
   - `_new_state` (`src/active_learning_sdk/engine.py:1101`)

Важно: после `__init__` движок может быть:

- "создан, но не настроен" (нет dataset_ref/label_schema в state),
- или "настроен ранее", но в текущем питон-процессе еще не прикреплены runtime-объекты (dataset/model/backend).

### 4.2) Engine.configure: первичная настройка (самая важная функция)

`ActiveLearningEngine.configure` (`src/active_learning_sdk/engine.py:483`) это "сборка пайплайна" и сохранение config в state.

Смотри по шагам:

1. Загружает state, если надо:
   - `_ensure_state_loaded` (`src/active_learning_sdk/engine.py:1118`)

2. Валидирует конфиги (каждый dataclass имеет validate):
   - `LabelSchema.validate` (`src/active_learning_sdk/configs.py`)
   - `AnnotationPolicy.validate` (`src/active_learning_sdk/configs.py`)
   - `SchedulerConfig.validate` (`src/active_learning_sdk/configs.py`)
   - `FingerprintConfig.validate` (`src/active_learning_sdk/configs.py`)
   - `SplitConfig.validate` (`src/active_learning_sdk/configs.py`)
   - `PrelabelConfig.validate` (`src/active_learning_sdk/configs.py`)
   - `LabelBackendConfig.validate` (`src/active_learning_sdk/configs.py`)

3. Приводит dataset к `DatasetProvider`:
   - `_coerce_dataset` (`src/active_learning_sdk/engine.py:1155`)
   - если это DataFrame/path, используется `DataFrameDatasetProvider` (`src/active_learning_sdk/dataset/provider.py:33`)

4. Считает fingerprint датасета:
   - `DatasetFingerprinter.fingerprint` (`src/active_learning_sdk/dataset/fingerprint.py:24`)
   - затем сравнивает с уже сохраненным fingerprint (если проект уже был настроен).
   - если mismatch -> `DatasetMismatchError` (`src/active_learning_sdk/exceptions.py`)

5. Проверяет модельный адаптер:
   - `inspect_model_capabilities` (`src/active_learning_sdk/adapters/base.py:45`)
   - `_validate_model_capabilities` (`src/active_learning_sdk/engine.py:1184`)
   - сейчас MVP требует: `predict_proba`, `fit`, `evaluate`.

6. Инициализирует кеши:
   - `_init_caches` (`src/active_learning_sdk/engine.py:1190`)
   - store на диске `JsonlDiskCacheStore` (`src/active_learning_sdk/cache.py:61`) или in-memory (`InMemoryCacheStore`).

7. Инициализирует backend разметки:
   - если backend объект передан напрямую, использует его.
   - иначе вызывает `build_label_backend` (`src/active_learning_sdk/backends/base.py:99`)
   - для Label Studio создается `LabelStudioBackend` (`src/active_learning_sdk/backends/label_studio.py:11`)
   - затем `backend.ensure_ready(label_schema)`:
     - `LabelStudioBackend.ensure_ready` (`src/active_learning_sdk/backends/label_studio.py:29`)

8. Создает `AnnotationAggregator`:
   - `AnnotationAggregator` (`src/active_learning_sdk/annotation.py:10`)

9. Создает scheduler со встроенными стратегиями:
   - `StrategyScheduler.__init__` (`src/active_learning_sdk/engine.py:196`)
   - и стратегии из `src/active_learning_sdk/strategies/uncertainty.py`.

10. Сохраняет все это в state:
    - state.dataset_ref, state.*_config, splits, sample_status
    - `_touch_state` (`src/active_learning_sdk/engine.py:1110`)
    - `_save_state` (`src/active_learning_sdk/engine.py:1114`) -> `JsonFileStateStore.save_atomic` (`src/active_learning_sdk/state/store.py:163`)

Если тебе нужно понять "что именно SDK гарантирует при перезапуске", то смотри структуру `ProjectState`:

- `ProjectState` (`src/active_learning_sdk/state/store.py:43`)
- `RoundState` (`src/active_learning_sdk/state/store.py:25`)

Там хранится то, что делает цикл идемпотентным: task_ids, статусы, выбранные sample_ids, snapshot scheduler.

### 4.3) Engine.attach_runtime: как открыть существующий workdir в новом процессе

`ActiveLearningEngine.attach_runtime` (`src/active_learning_sdk/engine.py:626`) это "ре-подключение живых объектов" к уже сохраненному конфигу.

Зачем нужно:

- В `state.json` лежат конфиги и история.
- Но объекты Python (модель, provider, backend) между процессами не сохраняются.

Что делает attach_runtime:

1. Проверяет, что проект был сконфигурирован (есть dataset_ref/label_schema).
2. Снова приводит dataset к provider (`_coerce_dataset`) и сверяет fingerprint с сохраненным.
3. Привязывает model, пересоздает label_schema/policy из сохраненного state.
4. Инициализирует кеши по сохраненному `cache_config`.
5. Инициализирует scheduler по сохраненному `scheduler_config`.
6. Backend: или берется переданный, или собирается из сохраненного `label_backend_config`.

### 4.4) Engine.run: блокирующий цикл

`ActiveLearningEngine.run` (`src/active_learning_sdk/engine.py:730`) это "петля".

Упрощенно:

1. Проверяет stop criteria (`StopCriteria`, `src/active_learning_sdk/configs.py`).
2. Пока не пора остановиться:
   - вызывает `run_step`.
   - если `run_step` вернул `NOOP`, sleep и продолжает.

Главное: run() не содержит бизнес-логики шагов. Это просто "крутилка", которая дергает `run_step`.

### 4.5) Engine.run_step: state machine

`ActiveLearningEngine.run_step` (`src/active_learning_sdk/engine.py:786`) определяет следующий шаг и выполняет его.

Схема:

1. Берет активный раунд или создает новый:
   - `_get_or_create_active_round` (`src/active_learning_sdk/engine.py:1263`)
2. Определяет следующий шаг по RoundStatus:
   - `_next_step` (`src/active_learning_sdk/engine.py:1286`)
3. Делает один из приватных шагов:
   - `_step_select` (`src/active_learning_sdk/engine.py:1305`)
   - `_step_push` (`src/active_learning_sdk/engine.py:1349`)
   - `_step_wait` (`src/active_learning_sdk/engine.py:1385`)
   - `_step_pull` (`src/active_learning_sdk/engine.py:1407`)
   - `_step_train_eval` (`src/active_learning_sdk/engine.py:1434`)
   - `_step_update` (`src/active_learning_sdk/engine.py:1489`)

Возвращает `StepResult` (`src/active_learning_sdk/engine.py:361`) чтобы интеграции могли логировать/показывать прогресс.

## 5) Что делает каждый шаг раунда (где смотреть и что реализовано)

### 5.1) SELECT: выбор примеров для разметки

`_step_select` (`src/active_learning_sdk/engine.py:1305`)

Что делает:

1. Собирает pool из UNLABELED ids из `state.sample_status`.
2. Если pool пустой:
   - переводит round в DONE
   - кидает `StopCriteriaReached` (`src/active_learning_sdk/exceptions.py`) чтобы `run()` завершился.
3. Создает `SelectionContext` (`src/active_learning_sdk/engine.py:56`):
   - внутри есть provider, model, caches, список labeled_ids, last_metrics.
4. Вызывает scheduler:
   - `StrategyScheduler.select_batch` (`src/active_learning_sdk/engine.py:214`)
5. Сохраняет:
   - `round_state.selected_sample_ids`
   - `round_state.scheduler_snapshot`
   - ставит status SELECTED
   - сохраняет state на диск.

Что уже готово:

- Scheduler для single/mix/bandit/custom режимов реализован.
- Стратегии uncertainty реализованы:
  - `EntropyStrategy.select` (`src/active_learning_sdk/strategies/uncertainty.py:24`)
  - `MarginStrategy.select` (`src/active_learning_sdk/strategies/uncertainty.py:61`)
  - `LeastConfidenceStrategy.select` (`src/active_learning_sdk/strategies/uncertainty.py:45`)
  - `RandomStrategy.select` (`src/active_learning_sdk/strategies/uncertainty.py:13`)

Что пока заглушка:

- `KCenterGreedyStrategy.select` кидает `NotImplementedError` (`src/active_learning_sdk/strategies/uncertainty.py:82`).

### 5.2) PUSH: отправка задач в backend разметки

`_step_push` (`src/active_learning_sdk/engine.py:1349`)

Что делает:

1. Если `round_state.task_ids` уже есть, это означает "уже пушили", шаг идемпотентен.
2. Берет выбранные sample_ids и получает `DataSample`:
   - `provider.get_samples(...)` (`DatasetProvider.get_samples` в `src/active_learning_sdk/dataset/provider.py:26`)
3. Если включен prelabeling (`state.prelabel_config.enable`), делает prelabels:
   - `_make_prelabels` (`src/active_learning_sdk/engine.py:1513`)
4. Вызывает backend:
   - `label_backend.push_round(...)` (протокол в `src/active_learning_sdk/backends/base.py:32`)
5. Сохраняет `task_ids` (sample_id -> backend task_id), status PUSHED, сохраняет state.

Что важно:

- Идемпотентность завязана на сохранение `task_ids` в state.
- Если процесс упадет после push, то при повторном запуске `_step_push` увидит `task_ids` и не будет пушить заново.

Что пока заглушка:

- Label Studio backend реально не реализован:
  - `LabelStudioBackend.push_round` (`src/active_learning_sdk/backends/label_studio.py:35`) -> `NotImplementedError`
  - `poll_round` (`src/active_learning_sdk/backends/label_studio.py:45`) -> `NotImplementedError`
  - `pull_round` (`src/active_learning_sdk/backends/label_studio.py:50`) -> `NotImplementedError`

То есть "поток" в MVP сейчас скелетный, backend нужно дописать, чтобы цикл реально работал с LS.

### 5.3) WAIT: ожидание разметки (polling)

`_step_wait` (`src/active_learning_sdk/engine.py:1385`)

Что делает:

1. Если round был PUSHED, переводит его в WAITING и сохраняет.
2. Вызывает backend.poll_round(...):
   - `label_backend.poll_round` (протокол в `src/active_learning_sdk/backends/base.py:32`)
3. Если прогресс говорит "всё готово", переводит round в READY_TO_PULL.
4. Возвращает `RoundProgress` (тип в `src/active_learning_sdk/backends/base.py:19`).

Что пока заглушка:

- В Label Studio backend polling не реализован.

### 5.4) PULL: забрать аннотации и применить policy

`_step_pull` (`src/active_learning_sdk/engine.py:1407`)

Что делает:

1. Вызывает backend.pull_round(...):
   - backend возвращает `annotations: sample_id -> List[AnnotationRecord]` (`src/active_learning_sdk/backends/base.py:27`)
2. Для каждого sample_id применяет aggregator:
   - `AnnotationAggregator.resolve` (`src/active_learning_sdk/annotation.py:17`)
3. Обновляет state по результату:
   - sample_status: LABELED/NEEDS_REVIEW/UNLABELED
   - sample_labels для LABELED
4. Пишет `round_state.resolved` и переводит раунд в PULLED.

Что уже готово:

- Aggregator реализован (режимы latest/first/majority/consensus).

### 5.5) TRAIN_EVAL: дообучение и метрики

`_step_train_eval` (`src/active_learning_sdk/engine.py:1434`)

Что делает:

1. Берет split ids из `state.splits` (создаются при configure):
   - `_resolve_splits` (`src/active_learning_sdk/engine.py:1207`)
2. Из train/val выбирает только LABELED.
3. Из provider забирает тексты и labels.
4. Вызывает model.fit(...) и затем model.evaluate(...) (если есть val).
5. Записывает `MetricRecord` в `state.metrics_history`:
   - `MetricRecord` (`src/active_learning_sdk/types.py`)
6. Если у модели нет стабильного `get_model_id()`, чистит кеш.
7. Ставит round status TRAINED.

Где контракт модели:

- `TextClassificationAdapter` (`src/active_learning_sdk/adapters/base.py:20`).

Что важно для джуна:

- Если у тебя модель "чистая sklearn", ты можешь сделать адаптер, который просто вызывает `predict_proba`, `fit`, `score/metrics`.
- Нельзя запускать цикл без `fit` и `evaluate` (MVP проверяет это в `_validate_model_capabilities`).

### 5.6) UPDATE: reward и обновление scheduler state

`_step_update` (`src/active_learning_sdk/engine.py:1489`)

Что делает:

1. Считает reward как дельту метрики:
   - `_compute_reward` (`src/active_learning_sdk/engine.py:1523`)
2. Вызывает `scheduler.update_reward(...)`:
   - `StrategyScheduler.update_reward` (`src/active_learning_sdk/engine.py:297`)
3. Ставит round status DONE.

Сейчас bandit-логика минимальная, но схема state предусмотрена.

## 6) Где хранится state и почему он важен

State это "память проекта", он лежит в `workdir/state.json`.

Схема:

- `ProjectState` (`src/active_learning_sdk/state/store.py:43`)
- `RoundState` (`src/active_learning_sdk/state/store.py:25`)
- `DatasetRef` (`src/active_learning_sdk/state/store.py:14`)

Как пишется:

- `JsonFileStateStore.save_atomic` (`src/active_learning_sdk/state/store.py:163`)
- внутри `atomic_write_text` (`src/active_learning_sdk/utils.py:28`) делает запись через temp file + `os.replace`.

Это важно для "не создавать дубли задач" при падениях.

## 7) Где "готово", а где "нужно реализовать"

Если ты путаешься, вот простой критерий:

- Если функция возвращает значение и не содержит `raise NotImplementedError(...)`, она скорее всего "готова" (пусть и простая).
- Если там `NotImplementedError` или явные комментарии "scaffold", значит это точка, которую надо доделать.

Явные заглушки сейчас:

- Label Studio backend:
  - `src/active_learning_sdk/backends/label_studio.py:35` (`push_round`)
  - `src/active_learning_sdk/backends/label_studio.py:45` (`poll_round`)
  - `src/active_learning_sdk/backends/label_studio.py:50` (`pull_round`)
- Report generator:
  - `src/active_learning_sdk/report.py:12`
- HF adapter:
  - `HFSequenceClassifierAdapter.fit/evaluate` (`src/active_learning_sdk/adapters/huggingface.py:37`, `:40`)
- Diversity strategy placeholder:
  - `KCenterGreedyStrategy.select` (`src/active_learning_sdk/strategies/uncertainty.py:82`)
- SplitConfig.mode="column":
  - `_resolve_splits` кидает `NotImplementedError` (`src/active_learning_sdk/engine.py:1207`)

То есть MVP по сути "каркас цикла" + реальные стратегии uncertainty + реальный state/caching/fingerprint.

## 8) Как не написать дубликаты и куда добавлять новый код

Типичные задачи и "правильное место":

- Реализовать интеграцию с Label Studio:
  - только в `src/active_learning_sdk/backends/label_studio.py`
  - engine уже умеет вызывать backend, трогать engine не надо, пока контракт соблюден.

- Добавить новую стратегию отбора:
  - реализовать класс с `.name` и `.select(...)` по протоколу `SamplingStrategy` (`src/active_learning_sdk/strategies/base.py:9`)
  - зарегистрировать через `project.register_strategy(...)` (`src/active_learning_sdk/project.py:90`, `engine.register_strategy` `src/active_learning_sdk/engine.py:711`).

- Добавить поддержку другого источника данных:
  - реализовать свой `DatasetProvider` (`src/active_learning_sdk/dataset/provider.py:17`)
  - engine уже принимает `dataset=DatasetProvider`.

- Улучшить кеш (LRU, SQLite, нормальный индекс):
  - `src/active_learning_sdk/cache.py` (stores)
  - engine использует CacheStore, менять engine не нужно.

- Изменить политику агрегации аннотаций:
  - `src/active_learning_sdk/annotation.py` (одна точка).

- Сделать нормальный HTML отчет:
  - `src/active_learning_sdk/report.py`

Если ты хочешь "сначала сделать MVP работающим", самое прямое:

1. Доделать `LabelStudioBackend.push_round/poll_round/pull_round`.
2. Оставить `ReportGenerator` заглушкой или сделать супер-простой отчет.

## 9) Мини-советы по навигации (практика)

Когда ты видишь публичный метод и хочешь понять "что он делает":

1. Открой метод в `project.py`.
2. Почти всегда увидишь `self._engine.<method>(...)`.
3. Иди в `engine.py` на соответствующую строку.
4. Внутри engine:
   - публичные методы (`configure/run/run_step/...`) обычно короткие,
   - настоящая логика в приватных `_step_*` и `_init_*`.

Полезные поиски:

- "где меняется статус раунда": ищи `round_state.status =` в `src/active_learning_sdk/engine.py`.
- "где пишется state": ищи `_save_state(`.
- "где дергается backend": ищи `.push_round(` / `.poll_round(` / `.pull_round(`.

## 10) Что дальше я бы делал на твоем месте (не код, а порядок)

Чтобы не разрасталось хаотично, бери изменения слоями:

1. Backend Label Studio (`src/active_learning_sdk/backends/label_studio.py`).
2. Минимальный репорт (хотя бы "таблица метрик" в HTML) в `src/active_learning_sdk/report.py`.
3. Тесты на идемпотентность раундов и state transitions (когда появится тестовый каркас).

