from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np


class LinearEpsilonGreedy:
    """Linear epsilon-greedy contextual bandit."""

    def __init__(
        self,
        n_actions: int,
        d: int,
        epsilon: float = 0.2,
        epsilon_min: float = 0.02,
        decay: float = 0.999,
        name: str = "LinEpsGreedy",
    ):
        self.n_actions = int(n_actions)
        self.d = int(d)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.decay = float(decay)
        self.name = name

        self.A = [np.eye(self.d) for _ in range(self.n_actions)]
        self.b = [np.zeros(self.d) for _ in range(self.n_actions)]
        self.t = 0
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.rewards_per_action = [[] for _ in range(self.n_actions)]

    def _q(self, action: int, x: np.ndarray) -> float:
        try:
            theta = np.linalg.solve(self.A[action], self.b[action])
            return float(theta @ x)
        except np.linalg.LinAlgError:
            return 0.0

    def _best_action(self, x: np.ndarray, mask: np.ndarray) -> int:
        vals = [self._q(i, x) if mask[i] else float("-inf") for i in range(self.n_actions)]
        return int(np.argmax(vals))

    def select_action(self, x: np.ndarray, mask: np.ndarray | None = None) -> int:
        x = np.atleast_1d(x).astype(float)
        if mask is None:
            mask = np.ones(self.n_actions, dtype=bool)
        valid = np.where(mask)[0]
        if len(valid) == 0:
            return 0

        if np.random.random() < self.epsilon:
            action = int(np.random.choice(valid))
        else:
            action = self._best_action(x, mask)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)
        return action

    def predict(self, x: np.ndarray, mask: np.ndarray | None = None) -> int:
        x = np.atleast_1d(x).astype(float)
        if mask is None:
            mask = np.ones(self.n_actions, dtype=bool)
        return self._best_action(x, mask)

    def update(self, x: np.ndarray, action: int, reward: float):
        x = np.atleast_1d(x).astype(float)
        a = int(action)
        r = float(reward)
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x
        self.t += 1
        self.action_counts[a] += 1
        self.rewards_per_action[a].append(r)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "LinearEpsilonGreedy":
        with Path(path).open("rb") as f:
            return pickle.load(f)
