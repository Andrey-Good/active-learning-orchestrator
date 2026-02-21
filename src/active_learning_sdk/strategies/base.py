from __future__ import annotations

"""
Sampling strategy interface.

A strategy decides which unlabeled sample_ids should be labeled next.

For juniors:
- A strategy does NOT talk to state.json directly.
- It receives a `SelectionContext` that provides access to the dataset and model.
"""

from typing import List, Protocol, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine import SelectionContext


class SamplingStrategy(Protocol):
    """
    Sampling strategy contract.

    Your custom strategy should:
    - define a unique `name`
    - implement `select(pool_ids, k, context)` and return a list of sample_ids

    Attributes:
        name (str):
            Where: used by `StrategyScheduler` to register and look up strategies.
            What: unique stable identifier for the strategy.
            Why: configs store strategy names (strings), so the project can resume after restart.
    """

    name: str

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> List[str]:
        ...
