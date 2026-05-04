from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from strategy_policies import BalancedStrategy


@dataclass
class TruncatedHybridPolicy:
    """Use action-level bandit for first K turns, then fallback to balanced strategy."""

    action_policy: Any
    k_turns: int

    def __post_init__(self):
        self.fallback = BalancedStrategy()

    def action(self, obs: np.ndarray, turn: int, action_mask: np.ndarray | None = None) -> int:
        if turn < self.k_turns:
            return int(self.action_policy.predict(obs, action_mask))
        return int(self.fallback.action(obs, turn, action_mask))
