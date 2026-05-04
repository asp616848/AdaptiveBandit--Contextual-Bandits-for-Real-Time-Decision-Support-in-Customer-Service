from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from simulation_core.config import ACTION_SPACE_7


@dataclass
class EpsilonGreedy:
    epsilon: float = 0.1

    def __post_init__(self):
        self.counts = np.zeros(len(ACTION_SPACE_7), dtype=float)
        self.values = np.zeros(len(ACTION_SPACE_7), dtype=float)

    def select(self, _context=None) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(len(ACTION_SPACE_7)))
        return int(np.argmax(self.values))

    def update(self, action: int, reward: float):
        self.counts[action] += 1.0
        n = self.counts[action]
        self.values[action] += (reward - self.values[action]) / n


@dataclass
class UCB1:
    def __post_init__(self):
        self.counts = np.zeros(len(ACTION_SPACE_7), dtype=float)
        self.values = np.zeros(len(ACTION_SPACE_7), dtype=float)
        self.t = 0

    def select(self, _context=None) -> int:
        self.t += 1
        for a in range(len(ACTION_SPACE_7)):
            if self.counts[a] == 0:
                return a
        bonus = np.sqrt((2 * np.log(max(self.t, 1))) / np.maximum(self.counts, 1e-9))
        return int(np.argmax(self.values + bonus))

    def update(self, action: int, reward: float):
        self.counts[action] += 1.0
        n = self.counts[action]
        self.values[action] += (reward - self.values[action]) / n


@dataclass
class ThompsonSampling:
    def __post_init__(self):
        self.alpha = np.ones(len(ACTION_SPACE_7), dtype=float)
        self.beta = np.ones(len(ACTION_SPACE_7), dtype=float)

    def select(self, _context=None) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, action: int, reward: float):
        # Reward is mapped to Bernoulli-like signal.
        r = 1.0 if reward > 0 else 0.0
        self.alpha[action] += r
        self.beta[action] += 1.0 - r
