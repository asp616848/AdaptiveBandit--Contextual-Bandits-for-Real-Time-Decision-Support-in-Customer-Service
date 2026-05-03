from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


STRATEGY_NAMES = {
    0: "info_first",
    1: "solve_first",
    2: "emotion_first",
    3: "escalate_early",
    4: "balanced",
}


def extract_strategy_context(state: dict) -> np.ndarray:
    """Context vector: [initial_sentiment, intent_confidence, complexity].

    sentiment is proxied as (1 - frustration) because explicit sentiment is not
    tracked in SupportEnv state.
    """
    frustration = float(state.get("frustration", 0.5))
    sentiment = float(np.clip(1.0 - frustration, 0.0, 1.0))
    intent_confidence = float(np.clip(state.get("information", 0.0), 0.0, 1.0))
    complexity = float(np.clip(state.get("difficulty", 0.5), 0.0, 1.0))
    return np.array([sentiment, intent_confidence, complexity], dtype=float)


@dataclass
class BaseStrategy:
    name: str

    def action(self, obs: np.ndarray, turn: int, action_mask: np.ndarray | None = None) -> int:
        raise NotImplementedError

    def _masked(self, action: int, action_mask: np.ndarray | None) -> int:
        if action_mask is None:
            return int(action)
        if 0 <= int(action) < len(action_mask) and bool(action_mask[int(action)]):
            return int(action)
        valid = np.where(action_mask)[0]
        return int(valid[0]) if len(valid) else 0


class InfoFirstStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("info_first")

    def action(self, obs: np.ndarray, turn: int, action_mask: np.ndarray | None = None) -> int:
        # Make this strategy slightly context-aware: if the user is already
        # frustrated, prioritize affective repair before asking for more info.
        frustration = float(obs[5]) if len(obs) > 5 else 0.5
        sentiment = 1.0 - frustration
        if sentiment < 0.35:
            a = 2  # AffectiveRepair
        elif turn < 3:
            a = 0  # AskInfo
        elif turn < 7:
            a = 1  # ProvideSolution
        else:
            a = 4  # Close
        return self._masked(a, action_mask)


class SolveFirstStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("solve_first")

    def action(self, obs: np.ndarray, turn: int, action_mask: np.ndarray | None = None) -> int:
        progress = float(obs[4]) if len(obs) > 4 else 0.0
        if turn < 2:
            a = 1  # ProvideSolution
        elif progress > 0.55:
            a = 4  # Close
        else:
            a = 1
        return self._masked(a, action_mask)


class EmotionFirstStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("emotion_first")

    def action(self, obs: np.ndarray, turn: int, action_mask: np.ndarray | None = None) -> int:
        frustration = float(obs[5]) if len(obs) > 5 else 0.5
        if turn < 2 or frustration > 0.45:
            a = 2  # AffectiveRepair
        elif turn < 5:
            a = 0  # AskInfo
        else:
            a = 1  # ProvideSolution
        return self._masked(a, action_mask)


class EscalateEarlyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("escalate_early")

    def action(self, obs: np.ndarray, turn: int, action_mask: np.ndarray | None = None) -> int:
        frustration = float(obs[5]) if len(obs) > 5 else 0.0
        if turn >= 3 and frustration > 0.25:
            a = 3  # Escalate
        elif turn < 3:
            a = 0
        else:
            a = 1
        return self._masked(a, action_mask)


class BalancedStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("balanced")

    def action(self, obs: np.ndarray, turn: int, action_mask: np.ndarray | None = None) -> int:
        information = float(obs[3]) if len(obs) > 3 else 0.0
        progress = float(obs[4]) if len(obs) > 4 else 0.0
        frustration = float(obs[5]) if len(obs) > 5 else 0.0

        if turn >= 3 and frustration > 0.7:
            a = 3
        elif frustration > 0.45:
            a = 2
        elif information < 0.4 and turn < 4:
            a = 0
        elif progress > 0.65 or turn > 8:
            a = 4
        else:
            a = 1
        return self._masked(a, action_mask)


def build_strategies() -> dict[int, BaseStrategy]:
    return {
        0: InfoFirstStrategy(),
        1: SolveFirstStrategy(),
        2: EmotionFirstStrategy(),
        3: EscalateEarlyStrategy(),
        4: BalancedStrategy(),
    }
