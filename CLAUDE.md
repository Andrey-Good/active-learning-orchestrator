# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Active Learning SDK — a stateful Python orchestrator for text classification active learning loops. It integrates ML models, sampling strategies, and annotation backends (Label Studio). Currently at **v0.1.0 (MVP)** stage, working toward the Phase 1 PRD target.

Python 3.9+. Apache 2.0 license.

## Build & Development

**No build system yet** — there is no `pyproject.toml`, `setup.py`, `Makefile`, test suite, or CI/CD. The project is pure source code under `src/active_learning_sdk/`.

To use the package in development, add `src/` to your Python path or install in editable mode once a build config is created.

**Core dependencies** are stdlib-only (`dataclasses`, `json`, `hashlib`, `pathlib`). Optional runtime deps: `pandas`, `torch`, `transformers`, `requests`, `xxhash`.

## Architecture

### Entry points and layering

```
User Code
  → ActiveLearningProject (project.py)     # thin facade, context manager
    → ActiveLearningEngine (engine.py)      # core state machine
      → Strategies, Backends, Adapters      # pluggable via protocols
```

`ActiveLearningProject` is the public API. `ActiveLearningEngine` contains all orchestration logic. Users never instantiate the engine directly.

### Round state machine (the core loop)

Each active learning round follows a strict state machine:

```
SELECTING → SELECTED → PUSHED → WAITING → READY_TO_PULL → PULLED → TRAINED → DONE
```

Each transition is one `run_step()` call. State is checkpointed to `workdir/state.json` after every step for crash safety. On resume, the engine picks up from the last saved status.

**Key invariant**: task IDs are stored in `RoundState.task_ids` after PUSH. On resume after crash, the engine reuses these IDs instead of creating duplicates.

### Protocol-based extensibility

All external integrations use `typing.Protocol`:
- **`TextClassificationAdapter`** (`adapters/base.py`) — model contract: `predict_proba`, `fit`, `evaluate` (required); `embed`, `save/load`, `get_model_id` (optional)
- **`LabelBackend`** (`backends/base.py`) — annotation UI: `ensure_ready`, `push_round`, `poll_round`, `pull_round`
- **`SamplingStrategy`** (`strategies/base.py`) — sample selection: `select(pool_ids, k, context)`
- **`DatasetProvider`** (`dataset/provider.py`) — data access: `iter_sample_ids`, `get_sample`, `schema`
- **`CacheStore`** / **`StateStore`** — storage abstractions

`inspect_model_capabilities()` in `adapters/base.py` does runtime capability detection on model adapters.

### Selection context

Strategies receive a read-only `SelectionContext` (defined in `engine.py`) instead of direct engine/state access. This contains dataset, model, caches, and metrics.

### Configuration

All config is via Python dataclasses in `configs.py` (not YAML/JSON files). Key configs: `LabelSchema`, `AnnotationPolicy`, `SchedulerConfig`, `CacheConfig`, `FingerprintConfig`, `StopCriteria`.

### State persistence

- `state/store.py` — `ProjectState` and `RoundState` dataclasses, serialized as JSON
- `state/lock.py` — file-based exclusive lock (`O_EXCL`) prevents concurrent runs
- `dataset/fingerprint.py` — dataset integrity via hashing (fast/strict/file modes, blake2b default)

### Annotation resolution

`annotation.py` — `AnnotationAggregator` resolves multi-annotator labels. Modes: `latest`, `first`, `majority` (with `min_agreement`), `consensus` (unanimous).

## Implementation status

**Fully implemented**: state machine, uncertainty strategies (entropy/margin/least-confidence/random), annotation aggregation, dataset fingerprinting, caching (memory + JSONL disk), file locking, DataFrame dataset provider, `LLMLabelBackend` (sync programmatic labeling).

**Scaffolds (raise `NotImplementedError`)**: `LabelStudioBackend` (all methods), `HFSequenceClassifierAdapter.fit/evaluate`, `ReportGenerator`, `KCenterGreedyStrategy`.

## PRD Phase 1 target architecture (`docs/PRD_Phase1_SDK.docx`)

The PRD defines the target state for this package (originally named `active_learning_core`). Key gaps between current code and PRD:

### State: JSON → SQLite
PRD requires SQLite-backed state (`sqlite_store.py`) with tables for `samples` (id, status, score, label, round_id), `rounds` (id, strategy, params, f1, sha256_weights), and `outbox_pending`. Current code uses JSON file state. The `StateStore` protocol (`ABCStore` in PRD) is the migration seam — implement `SqliteStateStore` behind the same interface.

### Strategies: uncertainty → TypiClust + BADGE
- **Round 0 — TypiClust**: cluster embeddings (all-MiniLM-L6-v2, dim=384) via FAISS k-means, select max-typicality sample per cluster. Replaces random cold start.
- **Round 1+ — BADGE**: hallucinated gradients → Random Projection (JL lemma, 7680→256 dims for OOM safety at N=1M) → FAISS k-means++. Captures uncertainty + diversity in one space with no manual α tuning.
- **Black-box fallback — B3AL**: Bayesian predictive kernels on output probabilities, for ONNX/sklearn models without gradient access.

### Calibration (not yet implemented)
- Temperature Scaling (Guo et al. 2017) — for neural models
- Platt Scaling — for ONNX/sklearn

### Stopping criteria: simple threshold → statistical bounds
- **PAC-Bayesian bounds** (primary): upper bound on generalization error delta; stop when expected accuracy gain is statistically indistinguishable from zero. No static val set needed.
- **Ranked Quasi-Sampling** (recall-critical): hypergeometric test for target recall τ with α=0.05. Requires a random-sample control pass before final stop (AL batches are not i.i.d.).

### Vector search: FAISS
FAISS in-memory with snapshot persistence (`faiss_snapshot.bin`). Used by TypiClust and BADGE. Incremental `add()` after each round, forced reindex when recall drops below threshold. Phase 2 migrates to pgvector.

### Inference: DataLoader batching
`hf_runner.py` (HuggingFace DataLoader with `num_workers`) and `onnx_runner.py` (ONNX Runtime). Replaces naive per-sample inference.

### Replay buffer: coreset selection
`replay/coreset.py` — stratified greedy centroid selection from `in_training` pool to prevent catastrophic forgetting on incremental `fit()`. Default `replay_ratio=0.3` of new batch size.

### Sample lifecycle (PRD)
```
new_unseen → scored → pending → sent → labeled → in_training
```
Current code uses a different status enum — needs alignment.

### Success criterion
Prove ≥80% annotation budget savings vs random baseline across 3 datasets within ~6 weeks.

## Documentation

README is bilingual (English + Russian). Additional Russian-language guides: `WALKTHROUGH_RU.md` (code walkthrough), `MVP_GUIDE_RU.md` (implementation guide), `CALLFLOW_TRACE_RU.md` (call flow analysis). Architecture diagram source at `docs/architecture.puml`. PRD documents in `docs/` (Phase 1 SDK, Phase 2 Infra).
