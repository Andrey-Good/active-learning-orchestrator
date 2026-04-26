import logging
import httpx
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class LabelStudioClient:
    """REST клиент для интеграции с Label Studio (Phase 1)."""

    def __init__(self, url: str, api_key: str, project_id: int):
        self.url = url.rstrip('/')
        self.project_id = project_id
        self.headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json"
        }

    def send_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Отправляет батч задач в Label Studio.
        Ожидает, что каждый сэмпл содержит 'idempotency_key' для защиты от дублей.
        """
        if not samples:
            return []

        logger.info(f"Sending {len(samples)} tasks to project {self.project_id}")

        # Формируем payload с защитой от дублей
        payload = []
        for sample in samples:
            payload.append({
                "data": {"text": sample["text"]},
                # Идемпотентность: критично для Crash Recovery и Transactional Outbox
                "meta": {"idempotency_key": sample["idempotency_key"], "sample_id": sample["id"]}
            })

        # Используем bulk API импорта
        endpoint = f"{self.url}/api/projects/{self.project_id}/import"

        with httpx.Client() as client:
            response = client.post(endpoint, headers=self.headers, json=payload)
            response.raise_for_status()

            result = response.json()
            logger.info(f"Successfully imported tasks. Task IDs: {result.get('task_ids', [])}")

            # Возвращаем информацию для обновления SQLite: pending -> sent
            return result.get('task_ids', [])

    def get_annotations(self, task_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Синхронно забирает аннотации. (Fallback / Polling для Phase 1).
        """
        # В реальной реализации здесь будет запрос GET /api/tasks?ids=...
        # Для базового SDK Phase 1 делаем выгрузку размеченных задач
        endpoint = f"{self.url}/api/projects/{self.project_id}/export?exportType=JSON"

        with httpx.Client() as client:
            response = client.get(endpoint, headers=self.headers)
            response.raise_for_status()

            export_data = response.json()

            # Фильтруем только те задачи, которые были в нашем запросе и имеют разметку
            labeled_tasks = []
            for task in export_data:
                if task["id"] in task_ids and task.get("annotations"):
                    # Извлекаем label и связываем с нашим внутренним sample_id
                    sample_id = task.get("meta", {}).get("sample_id")
                    annotation_result = task["annotations"][0]["result"]

                    labeled_tasks.append({
                        "task_id": task["id"],
                        "sample_id": sample_id,
                        "annotations": annotation_result
                    })

            return labeled_tasks