import pytest
from active_learning_core.core.session import ALSession
from active_learning_core.core.round import ALRound
from active_learning_core.strategies.typiclust import TypiClustStrategy
from active_learning_core.stopping.pac_bayesian import PACBayesStopping


def test_pool_initialization_in_db(temp_sqlite_store, sample_data_pool):
    """
    Проверяем, что пул корректно записался в SQLite с нужными статусами.
    """
    # Запрашиваем состояние напрямую из БД, а не из in-memory атрибутов
    unseen_count = temp_sqlite_store.count_samples_by_status("new_unseen")
    in_training_count = temp_sqlite_store.count_samples_by_status("in_training")

    assert unseen_count == 2, "Ожидаем 2 новых сэмпла в статусе 'new_unseen'"
    assert in_training_count == 1, "Ожидаем 1 сэмпл в 'in_training'"


def test_session_state_management(temp_sqlite_store, mock_label_studio_client):
    """
    Проверяем, что ALSession корректно инициализируется поверх SQLite.
    """
    session = ALSession(
        store=temp_sqlite_store,
        label_client=mock_label_studio_client
    )

    # Сессия должна прочитать последний раунд из БД (Crash Recovery)
    assert session.get_current_round_id() == 0
    assert session.get_status() == "initialized"


def test_al_round_zero_execution(
        temp_sqlite_store,
        sample_data_pool,
        mock_hf_runner,
        mock_faiss_index,
        mock_label_studio_client
):
    """
    Тестируем атомарный Round 0 (Холодный старт) с TypiClust.
    """
    session = ALSession(store=temp_sqlite_store, label_client=mock_label_studio_client)

    # Инициализируем раунд с конкретной стратегией для Round 0
    al_round = ALRound(
        session_id=session.id,
        store=temp_sqlite_store,
        pool=sample_data_pool,
        strategy=TypiClustStrategy(batch_size=2),
        stopping_criterion=PACBayesStopping(),
        inference_runner=mock_hf_runner
    )

    # Запускаем раунд
    round_result = al_round.execute()

    # Assertions по стейт-машине (PRD Жизненный цикл сэмплов)
    # Сэмплы должны были перейти из new_unseen -> scored -> pending -> sent
    sent_samples = temp_sqlite_store.get_samples_by_status("sent")
    assert len(sent_samples) == 2

    # Проверяем, что outbox был очищен (или содержит записи)
    outbox_count = temp_sqlite_store.get_outbox_pending_count()
    assert outbox_count == 0, "Outbox должен быть пустым после успешной отправки"

    assert round_result["status"] == "completed"