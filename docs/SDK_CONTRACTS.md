# Active Learning SDK Public Contracts

This document defines the public contract for `active_learning_sdk` as of Stage 1
release hardening. It is intentionally conservative: a name can be useful without
being stable, and implementation module paths are not guaranteed unless this file
says so.

## Stability Tiers

- Stable: supported for SDK users. Backward-compatible additions are allowed, but
  removing names, changing required arguments, changing persisted semantics, or
  changing exception categories requires a deliberate contract update.
- Provisional: usable extension surface, but names, keyword arguments, accepted
  configuration values, and exact payload shapes may still evolve before a stable
  release. Changes should be documented in release notes.
- Internal: implementation details. Users should not import, subclass, monkeypatch,
  or persist assumptions about these names.

The root package, `import active_learning_sdk`, is the stable import surface for
the names listed below. Concrete optional adapters must not be required merely to
import the root package.

## Stable Root Exports

The following names are the stable root package contract. They must remain
importable from `active_learning_sdk` and listed in `active_learning_sdk.__all__`.

<!-- PUBLIC_CONTRACT_STABLE_EXPORTS
ActiveLearningProject
SelectionContext
StepResult
StrategyScheduler
ActiveLearningError
ConfigurationError
DatasetMismatchError
ProjectLockedError
StateCorruptedError
InfrastructureError
LabelBackendError
StrategyError
ModelAdapterError
StopCriteriaReached
LabelSchema
AnnotationPolicy
SchedulerConfig
CacheConfig
FingerprintConfig
SplitConfig
LabelBackendConfig
PrelabelConfig
StopCriteria
DataSample
AnnotationRecord
ResolvedLabel
MetricRecord
SampleStatus
RoundStatus
StepKind
LabelBackend
RoundPushResult
RoundProgress
RoundPullResult
TextClassificationAdapter
ModelCapabilities
inspect_model_capabilities
CacheStore
InMemoryCacheStore
JsonlDiskCacheStore
PredictionCache
EmbeddingCache
SklearnTextClassifierAdapter
HFSequenceClassifierAdapter
END_PUBLIC_CONTRACT_STABLE_EXPORTS -->

## Project Facade

`ActiveLearningProject` is the stable user entrypoint. Its contract is to:

- configure a project from a dataset, model adapter, label schema, label backend,
  scheduler configuration, and optional policies;
- persist project state in the supplied `workdir`;
- resume an existing project when runtime-only objects are reattached;
- expose one-step and blocking run APIs through `run_step()` and `run()`;
- expose status, state inspection, round inspection, report generation, label
  export, split export, cache stats, cache clearing, and `close()`.

The public facade is stable. The engine object behind it, its private attributes,
and the exact internal state-machine implementation are internal.

## Configuration Dataclasses

The root-exported configuration dataclasses are stable:

- `LabelSchema`
- `AnnotationPolicy`
- `SchedulerConfig`
- `CacheConfig`
- `FingerprintConfig`
- `SplitConfig`
- `LabelBackendConfig`
- `PrelabelConfig`
- `StopCriteria`

Their field names and validation methods are part of the public contract. Adding
optional fields is allowed. Removing fields, changing defaults, or weakening
validation requires a contract update. Accepted enum-like string values are
stable only when documented in the dataclass docstrings or validation errors.

## Dataset Samples And Annotation Records

`DataSample`, `AnnotationRecord`, `ResolvedLabel`, and `MetricRecord` are stable
dataclasses for SDK boundary payloads.

- `DataSample.sample_id` must be stable and unique within a dataset. It is the
  persisted state key used for resume and idempotency.
- `DataSample.data` is the main sample payload. Text workflows expect a `"text"`
  value, but the mapping remains modality-neutral.
- `DataSample.meta` and `group_id` carry optional metadata used by splitting,
  reporting, and strategy logic.
- `AnnotationRecord` represents raw backend annotations before aggregation.
- `ResolvedLabel` represents the result of applying `AnnotationPolicy`.
- `MetricRecord` stores metric snapshots emitted by training or evaluation steps.

`SampleStatus`, `RoundStatus`, and `StepKind` enum values are stable state and API
vocabulary. New values may be added, but existing values must retain meaning.

## Selection Context, Strategies, And Scheduler

`SelectionContext` is the stable strategy-facing context. Strategies should use
its dataset, text, prediction, embedding, cache, labeled-id, metric, and
model-id accessors instead of reading project state directly.

`StrategyScheduler` is stable as a root import and as the scheduler used by the
project facade. Its exact built-in strategy registry and scheduling internals are
provisional. Custom strategies should follow the `SamplingStrategy` protocol in
`active_learning_sdk.strategies.base`, which is currently provisional because it
is not a root export.

The stable scheduler guarantee is that configured strategies return selected
sample ids and a scheduler snapshot suitable for persistence. The exact snapshot
schema is provisional unless it appears in persisted state compatibility tests.

`adaptive_uncertainty_diversity` is a pragmatic default candidate, not a universal
scientific optimum. For many-class text classification with very small labeled
coverage, it deliberately uses matched random exploration before switching to
diversity-prefiltered uncertainty. This avoids over-exploiting probability
support for only the classes already seen by the model. For smaller label spaces,
it retains the earlier guarded uncertainty/diversity behavior and later switches
to entropy.

## Caches

`CacheStore`, `InMemoryCacheStore`, `JsonlDiskCacheStore`, `PredictionCache`, and
`EmbeddingCache` are stable root exports.

The stable cache contract is:

- stores support `get`, `set`, `delete`, `stats`, and `clear`;
- prediction cache keys are scoped by model id, dataset fingerprint, and sample id;
- embedding cache keys are scoped by model id, dataset fingerprint, embedding
  configuration, and sample id;
- disk cache storage is single-process and single-writer.
- `stats()` separates current storage from historical activity:
  `stored_items`/`items` report entries currently present in the backing store,
  `current_reusable_items` reports entries matching the current model and data
  scope, `session_*` counters are reset when a store object is reopened, and
  `lifetime_*` counters are persisted for disk caches across reopen.
- When the model adapter does not expose a changed `get_model_id()` after
  training, the engine advances an internal cache epoch instead of reusing stale
  predictions. Old entries may remain in `stored_items`, while
  `current_reusable_items` becomes zero for the new epoch.
- Manual and automatic cache clears preserve disk-cache lifetime metadata and
  expose `last_clear_reason`, `last_clear_kind`, `last_cleared_items`, and
  `last_cleared_bytes`.
- Epoch invalidations that do not physically clear files are still observable
  through `lifetime_invalidations`, `last_invalidation_reason`,
  `last_invalidation_kind`, and `last_invalidated_at`.
- Custom cache stores with the older `stats()`/`clear()` signatures remain
  usable; SDK wrappers call the richer metadata contract when available and
  fall back to the legacy signatures instead of crashing.

Cache file formats and private key hashing details are internal.

## Label Backends

`LabelBackend`, `RoundPushResult`, `RoundProgress`, and `RoundPullResult` are
stable backend contracts.

Backends are called in this order: `ensure_ready()`, `push_round()`,
`poll_round()`, `pull_round()`, and `close()`. `push_round()` must be idempotent
for retries when possible, using `round_id` and `sample_id` as stable external
identifiers. Backend-specific payload fields are allowed, but users must not
depend on private concrete backend internals.

Concrete backend modules, managed Docker assets, and backend factory details are
provisional unless separately documented.

## Model Adapters And Capabilities

`TextClassificationAdapter`, `ModelCapabilities`, `inspect_model_capabilities`,
`SklearnTextClassifierAdapter`, and `HFSequenceClassifierAdapter` are stable root
exports. Concrete optional adapters are loaded lazily: importing
`active_learning_sdk` does not require scikit-learn, Transformers, or torch, but
accessing an optional adapter class requires its matching extra.

The minimal stable engine adapter contract for text classification is:

- `fit(texts, labels, **kwargs)` trains or updates the model;
- `evaluate(texts, labels)` returns numeric metrics.

Additional model capabilities are opt-in and validated against the configured
scheduler:

- `predict_proba(texts, batch_size=...)` returns one class-probability row per
  input text and is required by probability-based strategies and prelabeling;
- `predict_stochastic(texts, n=..., batch_size=...)`, when implemented, must
  return a sample-major probability cube shaped
  `[sample][stochastic_pass][label_probability]`; the first dimension must match
  the input text order and the second dimension must equal `n`;
- `predict_committee(texts, batch_size=...)`, when implemented, must return a
  sample-major probability cube shaped
  `[sample][committee_member][label_probability]`; each sample must have at
  least two committee members and a consistent member count;
- all probability rows returned by `predict_proba`, `predict_stochastic`, and
  `predict_committee` must be finite, non-negative, match `LabelSchema.labels`
  width when labels are configured, and sum to `1.0`;
- optional methods such as `get_model_id`, `embed`, `predict_logits`,
  `gradient_embed`, `predict_stochastic`, `predict_committee`, `save`, and
  `load` are discovered through `inspect_model_capabilities`.

Concrete adapter classes in optional extras are provisional and must not be
imported eagerly by the root package.

## Reports, State, And Resume Guarantees

At a high level, the SDK guarantees that project state is checkpointed in the
workdir, rounds are resumable, and task ids returned by label backends are used
to avoid duplicate pushes after retries. Dataset fingerprints and split
assignments protect against accidentally resuming a project with incompatible
data.

The stable user contract is exposed through `ActiveLearningProject` methods such
as `get_state()`, `status()`, `list_rounds()`, `get_round()`, `validate()`,
`generate_report()`, `export_labels()`, and `export_dataset_split()`. The exact
JSON state layout, report HTML structure, and private state-store classes are
internal unless covered by explicit compatibility tests.

## Error Taxonomy

All public SDK exceptions inherit from `ActiveLearningError`.

| Category | Exception | Meaning |
| --- | --- | --- |
| User/configuration | `ConfigurationError` | Invalid configuration, invalid payload shape, missing required runtime object, or unsupported mode. |
| User/configuration | `DatasetMismatchError` | Input data or split assignments do not match persisted project identity. |
| Infrastructure | `ProjectLockedError` | Another process owns the project lock or the lock cannot be acquired. |
| Infrastructure | `InfrastructureError` | Local infrastructure, filesystem, process, or managed service setup is unavailable or misconfigured. |
| Backend | `LabelBackendError` | Label backend communication, protocol, idempotency, or payload parsing failed. |
| Model adapter | `ModelAdapterError` | User model adapter failed or violated the adapter contract at runtime. |
| State corruption | `StateCorruptedError` | Persisted state cannot be loaded, parsed, validated, or safely resumed. |
| Strategy | `StrategyError` | Sampling strategy failed or returned invalid selections. |
| Stop criteria | `StopCriteriaReached` | Control-flow signal that the run loop stopped cleanly because stopping criteria were reached. |

## Provisional And Internal Surfaces

The following are provisional unless promoted in this document:

- concrete adapter module internals behind the stable root adapter classes;
- concrete backend modules such as Label Studio, simulator, LLM, and managed
  Docker backends;
- built-in strategy names, exact strategy scoring details, and scheduler snapshot
  payloads;
- dataset provider convenience classes and dataframe ingestion details;
- benchmark, audit, and release-hardening artifacts.

The following are internal:

- private attributes and methods;
- implementation module paths not named here as contracts;
- raw state-store internals and lock internals;
- report HTML structure and cache file encodings;
- any test-only helpers or audit-only fixtures.
