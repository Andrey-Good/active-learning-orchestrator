from typing import Dict, Any
from active_learning_core.inference.hf_runner import run_inference
from active_learning_core.strategies.badge import BADGEStrategy
from active_learning_core.strategies.typiclust import TypiClustStrategy


class ALRound:
    """Реализация атомарного цикла Active Learning Phase 1[cite: 13]."""

    def __init__(self, session_id: str, store: SQLiteStore, model: Any, round_number: int):
        self.session_id = session_id
        self.store = store
        self.model = model
        self.round_number = round_number

    def execute(self, budget: int) -> Dict[str, Any]:
        # 1. Инференс (DataLoader-батчинг для обхода GIL)
        unlabeled = self.store.get_samples_by_status("new_unseen")
        probs = run_inference(self.model, [s['text'] for s in unlabeled])

        # 2. Выбор стратегии: TypiClust (Round 0) или BADGE (Round 1+) [cite: 37, 40]
        if self.round_number == 0:
            strategy = TypiClustStrategy()
        else:
            # BADGE использует Random Projection для защиты от OOM [cite: 42, 43]
            strategy = BADGEStrategy()

        selected_ids = strategy.select(unlabeled, probs, budget)

        # 3. Регистрация в Outbox (подготовка к отправке в Label Studio) [cite: 35, 112]
        self.store.move_to_pending(selected_ids)

        # 4. Проверка критериев остановки (PAC-Bayesian Bounds) [cite: 51]
        should_stop = self.store.check_stopping_criteria()

        return {
            "round_id": self.round_number,
            "selected_count": len(selected_ids),
            "should_stop": should_stop
        }