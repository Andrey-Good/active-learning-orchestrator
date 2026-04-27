from __future__ import annotations

"""
Abstract storage contract for project state.

`ABCStore` is the migration seam between the current JSON file state and future
SQLite/Postgres-backed state.  The engine only needs `load()` and
`save_atomic()`, while transactional and sample-status helpers are provided for
storage engines that can update individual records efficiently.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

from ..types import SampleStatus

if TYPE_CHECKING:
    from .store import ProjectState


StatusLike = Union[SampleStatus, str]


def normalize_status(status: StatusLike) -> str:
    """Return the persisted string value for a sample status."""
    return status.value if isinstance(status, SampleStatus) else str(status)


class ABCStore(ABC):
    """Base contract for project-state stores.

    Implementations must make `save_atomic()` crash-safe. Transaction methods are
    no-ops for stores that do not support real transactions, but SQLite/Postgres
    stores should implement them with real BEGIN/COMMIT/ROLLBACK semantics.
    """

    @abstractmethod
    def load(self) -> ProjectState:
        """Load the complete project state."""
        raise NotImplementedError

    @abstractmethod
    def save_atomic(self, state: ProjectState) -> None:
        """Persist the complete project state atomically."""
        raise NotImplementedError

    @abstractmethod
    def begin_transaction(self) -> None:
        """Start a write transaction when the backend supports it."""
        raise NotImplementedError

    @abstractmethod
    def commit(self) -> None:
        """Commit the active write transaction."""
        raise NotImplementedError

    @abstractmethod
    def rollback(self) -> None:
        """Rollback the active write transaction."""
        raise NotImplementedError

    @abstractmethod
    def get_sample_status(self, sample_id: str) -> Optional[str]:
        """Return a sample status value, or None if the sample is unknown."""
        raise NotImplementedError

    @abstractmethod
    def set_sample_status(self, sample_id: str, status: StatusLike) -> None:
        """Persist a status for one sample."""
        raise NotImplementedError

    @abstractmethod
    def get_samples_by_status(self, status: StatusLike) -> List[str]:
        """Return sample IDs that currently have the given status."""
        raise NotImplementedError

    def set_samples_status(self, sample_ids: Sequence[str], status: StatusLike) -> None:
        """Convenience helper for stores that do not provide a batch override."""
        for sample_id in sample_ids:
            self.set_sample_status(sample_id, status)
