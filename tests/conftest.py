import os
import tempfile
import pytest
import numpy as np
from unittest.mock import MagicMock

from active_learning_core.state.sqlite_store import SQLiteStore
from active_learning_core.core.pool import Pool
from active_learning_core.label_studio.client import LabelStudioClient


# -----------------------------------------------------------------------------
# 1. State Management (SQLite)
# -----------------------------------------------------------------------------
@pytest.fixture
def temp_sqlite_store():
    """
    Создает изолированный SQLite-файл для тестов.
    Покрывает требования Phase 1: Один файл на диске[cite: 34].
    """
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    # Инициализация схемы: таблицы samples, rounds, outbox_pending
    store = SQLiteStore(db_path=path)
    store.initialize_schema()

    yield store

    # Очистка после тестов
    store.close()
    os.close(fd)
    os.remove(path)


# -----------------------------------------------------------------------------
# 2. Data Pool & Жизненный цикл сэмплов
# -----------------------------------------------------------------------------
@pytest.fixture
def sample_data_pool(temp_sqlite_store):
    """
    Пул сэмплов, привязанный к SQLite.
    По умолчанию статус 'new_unseen'.
    """
    pool = Pool(store=temp_sqlite_store)
    samples = [
        {"id": "txt-001", "text": "симптомы ОРВИ", "status": "new_unseen"},
        {"id": "txt-002", "text": "подозрение на пневмонию", "status": "new_unseen"},
        {"id": "txt-003", "text": "здоров", "status": "in_training"}  # Для теста replay buffer
    ]
    pool.add_samples(samples)
    return pool


# -----------------------------------------------------------------------------
# 3. Инференс (DataLoader-батчинг)
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_hf_runner(mocker):
    """
    Мок для inference/hf_runner.py.
    Возвращает матрицу вероятностей float32[N, num_classes].
    """
    runner_mock = mocker.patch("active_learning_core.inference.hf_runner.run_inference")

    # Имитация N=2, C=3 (3 класса)
    def fake_inference(model, pool_texts, batch_size=64, num_workers=4):
        N = len(pool_texts)
        return np.random.dirichlet(np.ones(3), size=N).astype(np.float32)

    runner_mock.side_effect = fake_inference
    return runner_mock


@pytest.fixture
def mock_badge_gradients():
    """
    Для тестирования BADGE (Round 1+).
    Генерирует сжатые эмбеддинги после Random Projection (JL lemma) 7680 -> 256 dims[cite: 42, 43].
    """

    def _generate_grads(N: int):
        return np.random.randn(N, 256).astype(np.float32)

    return _generate_grads


# -----------------------------------------------------------------------------
# 4. Label Studio Integration
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_label_studio_client(mocker):
    """
    Мок для label_studio/client.py.
    Имитирует фиксацию idempotency key при отправке задач.
    """
    client_mock = MagicMock(spec=LabelStudioClient)

    # Имитируем успешную отправку батча
    client_mock.send_batch.return_value = {
        "status": "success",
        "idempotency_key": "test-uuid-1234",
        "tasks_created": 2
    }

    return client_mock


# -----------------------------------------------------------------------------
# 5. FAISS Vector Store
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_faiss_index(mocker):
    """
    Мок для FAISS In-memory индекса, используемого в TypiClust и BADGE.
    """
    faiss_mock = mocker.patch("active_learning_core.strategies.typiclust.faiss")
    index_mock = MagicMock()
    # Возвращаем индексы центроидов
    index_mock.search.return_value = (np.array([[0.1]]), np.array([[1]]))
    faiss_mock.IndexFlatL2.return_value = index_mock
    return faiss_mock