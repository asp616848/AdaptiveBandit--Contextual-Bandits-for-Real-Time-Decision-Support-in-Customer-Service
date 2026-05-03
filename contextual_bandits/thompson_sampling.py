from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle

import numpy as np


@dataclass
class _ArmPosterior:
    d: int
    v: float = 0.5
    A: np.ndarray = field(default_factory=lambda: None)
    b: np.ndarray = field(default_factory=lambda: None)

    def __post_init__(self):
        if self.A is None:
            self.A = np.eye(self.d)
        if self.b is None:
            self.b = np.zeros(self.d)


class LinearThompsonSampling:
    """Linear Thompson Sampling contextual bandit with one posterior per action."""

    def __init__(self, n_actions: int, d: int, v: float = 0.5, name: str = "LinTS"):
        self.n_actions = int(n_actions)
        self.d = int(d)
        self.v = float(v)
        self.name = name
        self.arms = [_ArmPosterior(d=self.d, v=self.v) for _ in range(self.n_actions)]
        self.t = 0
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.rewards_per_action = [[] for _ in range(self.n_actions)]

    def _sample_value(self, arm: _ArmPosterior, x: np.ndarray) -> float:
        try:
            A_inv = np.linalg.inv(arm.A)
            mu = A_inv @ arm.b
            theta = np.random.multivariate_normal(mu, (arm.v ** 2) * A_inv)
            return float(theta @ x)
        except np.linalg.LinAlgError:
            return float(np.random.normal(0.0, 1.0))

    def _mean_value(self, arm: _ArmPosterior, x: np.ndarray) -> float:
        try:
            theta = np.linalg.solve(arm.A, arm.b)
            return float(theta @ x)
        except np.linalg.LinAlgError:
            return float("-inf")

    def select_action(self, x: np.ndarray, mask: np.ndarray | None = None) -> int:
        x = np.atleast_1d(x).astype(float)
        if mask is None:
            mask = np.ones(self.n_actions, dtype=bool)
        vals = [self._sample_value(arm, x) if mask[i] else float("-inf") for i, arm in enumerate(self.arms)]
        return int(np.argmax(vals))

    def predict(self, x: np.ndarray, mask: np.ndarray | None = None) -> int:
        x = np.atleast_1d(x).astype(float)
        if mask is None:
            mask = np.ones(self.n_actions, dtype=bool)
        vals = [self._mean_value(arm, x) if mask[i] else float("-inf") for i, arm in enumerate(self.arms)]
        return int(np.argmax(vals))

    def update(self, x: np.ndarray, action: int, reward: float):
        x = np.atleast_1d(x).astype(float)
        a = int(action)
        r = float(reward)
        self.arms[a].A += np.outer(x, x)
        self.arms[a].b += r * x
        self.t += 1
        self.action_counts[a] += 1
        self.rewards_per_action[a].append(r)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "LinearThompsonSampling":
        with Path(path).open("rb") as f:
            return pickle.load(f)
