from active_learning_core.state.sqlite_store import SQLiteStore
from .round import ALRound


class ALSession:
    """Управление жизненным циклом сессии обучения[cite: 11]."""

    def __init__(self, db_path: str, model_config: Dict[str, Any]):
        self.store = SQLiteStore(db_path)  # Всё состояние в одном файле [cite: 32]
        self.status = "initialized"

    def run_next_round(self, model: Any, budget: int) -> Dict[str, Any]:
        current_round_no = self.store.get_last_round_number() + 1

        # Создание раунда и выполнение
        al_round = ALRound(self.session_id, self.store, model, current_round_no)
        result = al_round.execute(budget)

        # Фиксация версии раунда (иммутабельность и SHA256 весов) [cite: 133, 137]
        self.store.commit_round(result)

        if result["should_stop"]:
            self.status = "stopped_by_criteria"
        else:
            self.status = "waiting_for_labels"

        return result