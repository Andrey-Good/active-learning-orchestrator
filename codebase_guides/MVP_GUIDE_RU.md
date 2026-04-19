# MVP guide (RU): какие файлы трогать и что в них делать

Этот файл — “карта MVP” для разработчиков, которые впервые видят репозиторий.
Цель: не утонуть в структуре и сразу понимать:

- **где** лежит нужная логика,
- **что** именно надо реализовать/проверить,
- **зачем** это нужно в MVP.

## Что считается MVP в этом репозитории

MVP должен позволять пользователю:

1) передать датасет + модель,
2) выбрать стратегию семплирования (хотя бы `random`, лучше ещё uncertainty),
3) запустить цикл Active Learning,
4) отправить выбранные примеры в **Label Studio (LS)**,
5) дождаться разметки и забрать её обратно,
6) дообучить модель,
7) получить:
   - “обновлённую” модель (та же ссылка на объект модели, но после `fit()`),
   - размеченные данные (через `export_labels()` / `get_state()`).

## Самый короткий пользовательский API (что вызывает юзер)

Файл: `src/active_learning_sdk/project.py`

Обычно пользователь делает так:

1. `project = ActiveLearningProject(project_name, workdir=...)`
2. `project.configure(dataset=..., model=..., label_schema=..., label_backend_config=..., scheduler_config=...)`
3. `project.run(...)` **или** многократно `project.run_step(...)`

А чтобы “забрать результаты”:

- `project.export_labels(...)`
- `project.export_dataset_split(...)`
- `project.get_state()`

## MVP state-machine (внутренняя логика цикла)

Файл: `src/active_learning_sdk/engine.py`

Цикл устроен как state-machine по раундам:

`SELECT -> PUSH -> WAIT -> PULL -> TRAIN_EVAL -> UPDATE -> (следующий раунд)`

Где:

- **SELECT**: выбираем `k` id из пула (стратегия).
- **PUSH**: создаём задачи в LS.
- **WAIT**: ждём, пока аннотаций достаточно (по политике).
- **PULL**: забираем аннотации и агрегируем в итоговую метку.
- **TRAIN_EVAL**: обучаем/оцениваем модель.
- **UPDATE**: обновляем состояние/награду шедулера и сохраняем checkpoint.

Весь прогресс сохраняется в `workdir/state.json`, чтобы можно было продолжить после перезапуска.

---

# Файлы MVP: что использовать “как есть”, а что дописывать

Ниже — список **всех файлов пакета**, которые участвуют в MVP-потоке (включая те,
которые уже готовы, но их важно знать и “не забыть”).

Формат каждого пункта:

- **Статус**: `MVP: implement` / `MVP: tweak` / `MVP: use as-is` / `Not needed for MVP`
- **Что делать**: кратко
- **Ключевые точки**: важные классы/методы

## 1) Публичный API (то, что видит пользователь)

### `src/active_learning_sdk/__init__.py`
- Статус: `MVP: use as-is`
- Что делать: экспортирует публичные классы/конфиги/ошибки.
- Ключевые точки: `ActiveLearningProject`, `LabelSchema`, `SchedulerConfig`, `LabelBackendConfig`, …

### `src/active_learning_sdk/project.py`
- Статус: `MVP: tweak (редко)` / чаще `use as-is`
- Что делать:
  - держать фасад тонким (не раздувать бизнес-логикой),
  - добавлять только удобные пользовательские методы (если понадобится).
- Ключевые точки (что пользователь вызывает):
  - `ActiveLearningProject.configure()`
  - `ActiveLearningProject.run()` / `ActiveLearningProject.run_step()`
  - `ActiveLearningProject.export_labels()`
  - `ActiveLearningProject.get_state()`, `status()`, `list_rounds()`, `get_round()`

## 2) Engine (главная логика цикла)

### `src/active_learning_sdk/engine.py`
- Статус: `MVP: tweak/finish`
- Что делать (MVP):
  - убедиться, что весь state-machine проходит полный путь без ручных вмешательств,
  - корректно пишет/читает `state.json` (resume),
  - корректно вызывает backend и модель.
- Ключевые точки:
  - `ActiveLearningEngine.configure()`
  - `ActiveLearningEngine.run()` / `ActiveLearningEngine.run_step()`
  - `ActiveLearningEngine.attach_runtime()` (когда открыли существующий `workdir`)
  - `SelectionContext` (то, что получают стратегии)
  - `StrategyScheduler` (выбор стратегии и батча)
  - Экспорт/результаты: `export_labels()`, `export_dataset_split()`, `get_state()`

Важно для MVP:
- Сейчас engine требует, чтобы модель-адаптер имел `predict_proba()`, `fit()`, `evaluate()`
  (см. `_validate_model_capabilities()`).

## 3) Backend (Label Studio) — главный кусок работы MVP

### `src/active_learning_sdk/backends/base.py`
- Статус: `MVP: use as-is` (контракты), иногда `tweak`
- Что делать:
  - редко меняется: это “протокол” взаимодействия engine ↔ backend.
  - если LS требует дополнительных данных/форматов — лучше расширять конфиги/LS backend,
    а не ломать контракт engine.
- Ключевые точки:
  - `LabelBackend` (Protocol): `ensure_ready()`, `push_round()`, `poll_round()`, `pull_round()`, `close()`
  - `RoundPushResult`, `RoundProgress`, `RoundPullResult`
  - `build_label_backend()` (фабрика по `LabelBackendConfig`)

### `src/active_learning_sdk/backends/label_studio.py`
- Статус: `MVP: implement` (самое важное)
- Что делать (MVP минимум):
  1) `ensure_ready(label_schema)`:
     - проверить доступность LS (base_url/token),
     - создать/переиспользовать проект,
     - настроить интерфейс разметки под `LabelSchema`,
     - сохранить/вернуть `project_ref` (id/urls) для дебага.
  2) `push_round(round_id, samples, prelabels=None)`:
     - создать задачи в LS **идемпотентно** (повторный вызов не должен плодить дубликаты),
     - вернуть `RoundPushResult(task_ids=...)`, где `task_ids`: sample_id -> ls_task_id.
  3) `poll_round(round_id, task_ids, policy)`:
     - проверить прогресс по задачам,
     - решить “готово ли достаточно аннотаций” по `AnnotationPolicy`,
     - вернуть `RoundProgress(total=..., done=..., ready_sample_ids=...)`.
  4) `pull_round(round_id, task_ids)`:
     - получить аннотации из LS,
     - привести их к формату `AnnotationRecord` (см. `types.py`),
     - вернуть `RoundPullResult(annotations=...)` как sample_id -> list[AnnotationRecord].

Подсказка:
- Реально удобнее сначала сделать MVP на “самом простом” формате:
  - 1 аннотация = 1 label
  - без сложных structured outputs
  - `AnnotationRecord.value` = строка/число/список (как решите)

### `src/active_learning_sdk/annotation.py`
- Статус: `MVP: use as-is` (или `tweak`, если LS формат сложнее)
- Что делать:
  - агрегировать 1..N аннотаций в одну финальную метку (`ResolvedLabel`).
  - если в LS бывают конфликты/несогласие — правильно отдавать `NEEDS_REVIEW`.
- Ключевые точки:
  - `AnnotationAggregator.resolve(sample_id, annotations)`

## 4) Стратегии семплирования (эвристики)

### `src/active_learning_sdk/strategies/base.py`
- Статус: `MVP: use as-is`
- Что делать: контракт стратегий.
- Ключевые точки:
  - `SamplingStrategy.name`
  - `SamplingStrategy.select(pool_ids, k, context)`

### `src/active_learning_sdk/strategies/uncertainty.py`
- Статус: `MVP: use as-is`
- Что делать:
  - для MVP уже есть готовые стратегии: `random`, `entropy`, `least_confidence`, `margin`
  - `KCenterGreedyStrategy` — scaffold (для эмбеддингов), в MVP можно не включать.
- Ключевые точки:
  - `RandomStrategy.select(...)`
  - `EntropyStrategy.select(...)`
  - `LeastConfidenceStrategy.select(...)`
  - `MarginStrategy.select(...)`

Важно:
- Uncertainty стратегии требуют, чтобы модель поддерживала `predict_proba()`.

## 5) Модель (adapter layer)

### `src/active_learning_sdk/adapters/base.py`
- Статус: `MVP: use as-is` (контракт)
- Что делать:
  - это не “готовая модель”, а интерфейс, который должен реализовать пользователь.
- Ключевые точки:
  - `TextClassificationAdapter` (Protocol): `predict_proba`, `fit`, `evaluate` (+ опционально `get_model_id`, `embed`)
  - `inspect_model_capabilities()`

### `src/active_learning_sdk/adapters/huggingface.py`
- Статус: `Not needed for MVP` (если вы не хотите поддерживать transformers прямо сейчас)
- Что делать:
  - это заготовка адаптера для HF, полезно позже.
- Ключевые точки:
  - `HFSequenceClassifierAdapter.predict_proba()`

## 6) Датасет (provider layer)

### `src/active_learning_sdk/dataset/provider.py`
- Статус: `MVP: use as-is` (обычно хватает)
- Что делать:
  - дать engine единый способ получать `DataSample` по `sample_id`.
  - сейчас поддерживается:
    - `DatasetProvider` (пользовательский),
    - `pandas.DataFrame` через `DataFrameDatasetProvider`,
    - path к CSV/Parquet (если есть pandas).
- Ключевые точки:
  - `DatasetProvider.iter_sample_ids()`
  - `DatasetProvider.get_sample(sample_id)`
  - `DataFrameDatasetProvider`

### `src/active_learning_sdk/dataset/fingerprint.py`
- Статус: `MVP: use as-is`
- Что делает:
  - защищает от случайной подмены датасета при использовании того же `workdir`.
- Ключевые точки:
  - `DatasetFingerprinter.fingerprint(provider)`

## 7) Состояние/чекпоинты/лок

### `src/active_learning_sdk/state/store.py`
- Статус: `MVP: use as-is`
- Что делает:
  - описывает форматы `state.json` и атомарную запись.
- Ключевые точки:
  - `ProjectState`, `RoundState`, `DatasetRef`
  - `JsonFileStateStore.load()` / `save_atomic()`

### `src/active_learning_sdk/state/lock.py`
- Статус: `MVP: use as-is`
- Что делает:
  - защищает `workdir` от одновременного запуска двумя процессами.
- Ключевые точки:
  - `ProjectLock.acquire()` / `release()`

## 8) Конфиги/типы/ошибки — “склейка” всего MVP

### `src/active_learning_sdk/configs.py`
- Статус: `MVP: tweak (по необходимости)`
- Что делать:
  - тут датаклассы конфигов, которые пользователь передаёт в `configure()`.
  - для LS почти всегда приходится уточнять поля `LabelBackendConfig` (base_url/token/project_id и т.п.)
    и/или поля `LabelSchema` (как именно описывается разметка).
- Ключевые точки:
  - `LabelSchema`
  - `LabelBackendConfig`
  - `SchedulerConfig`
  - `AnnotationPolicy`
  - `SplitConfig`, `StopCriteria` (условия остановки)

### `src/active_learning_sdk/types.py`
- Статус: `MVP: tweak (по необходимости)`
- Что делать:
  - типы данных, которыми обмениваются engine ↔ backend ↔ state.
  - если LS отдаёт сложную структуру, вы либо:
    - кладёте её в `AnnotationRecord.value/details`,
    - либо расширяете типы.
- Ключевые точки:
  - `DataSample`
  - `AnnotationRecord`
  - `ResolvedLabel`
  - `RoundStatus`, `SampleStatus`

### `src/active_learning_sdk/exceptions.py`
- Статус: `MVP: use as-is`
- Что делать:
  - единые ошибки SDK, чтобы пользователю было проще ловить исключения.

## 9) Кэш/утилиты/репорт — вторично для MVP, но используется

### `src/active_learning_sdk/cache.py`
- Статус: `MVP: use as-is`
- Что делает:
  - ускоряет выборку (predict_proba/embed) и помогает при resume.
- Ключевые точки:
  - `CacheConfig` (включение/персистентность)
  - `PredictionCache`, `EmbeddingCache`
  - `InMemoryCacheStore`, `JsonlDiskCacheStore`

### `src/active_learning_sdk/utils.py`
- Статус: `MVP: use as-is`
- Что делает:
  - атомарные записи, сериализация датаклассов и прочие мелочи.

### `src/active_learning_sdk/report.py`
- Статус: `Not needed for MVP`
- Почему:
  - `ReportGenerator.generate_html()` сейчас `NotImplementedError`.
  - Для MVP “получить размеченные данные” обычно хватает `export_labels()`.

---

# “Что точно нужно реализовать” — короткий список задач MVP

1) Реализовать LS backend:
   - `src/active_learning_sdk/backends/label_studio.py`
     - `ensure_ready()`
     - `push_round()`
     - `poll_round()`
     - `pull_round()`

2) Убедиться, что пользовательская модель соответствует контракту:
   - `src/active_learning_sdk/adapters/base.py` (контракт)
   - (внешний код пользователя) реализует `predict_proba()`, `fit()`, `evaluate()`

3) Проверить, что данные от backend корректно агрегируются:
   - `src/active_learning_sdk/annotation.py`
   - формат `AnnotationRecord` в `src/active_learning_sdk/types.py`

4) Документация для команды:
   - `README.md` (коротко: как запустить MVP, какие части scaffold)
   - этот файл (`MVP_GUIDE_RU.md`) как “карта”.

---

# Acceptance criteria (как понять, что MVP готов)

Сценарий “с нуля”:

1) Создать `ActiveLearningProject(..., workdir=...)`
2) `configure(...)` с:
   - датасетом (DataFrame или DatasetProvider)
   - моделью-адаптером
   - стратегией `random` (или uncertainty)
   - LS backend config
3) `run(budget=..., batch_size=...)` проходит хотя бы 1 полный раунд без ручного вмешательства:
   - в LS появились задачи,
   - появились аннотации,
   - SDK их забрал и сохранил в `state.json`,
   - модель прошла `fit()` и `evaluate()`.
4) `export_labels(...)` создаёт файл с размеченными данными.

