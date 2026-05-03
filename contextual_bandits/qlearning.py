"""Q-Learning implementations for SupportEnv."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any
import pickle
from pathlib import Path


@dataclass
class QLearning:
    """Tabular Q-Learning with function approximation."""
    
    n_actions: int
    n_state_bins: int = 32  # Discretize continuous state into bins
    alpha: float = 0.1  # Learning rate
    gamma: float = 0.99  # Discount factor
    epsilon: float = 0.1  # Exploration rate
    
    Q: dict = field(default_factory=dict)
    training_steps: int = 0
    
    def _state_to_idx(self, obs: np.ndarray) -> tuple:
        """Discretize continuous observation into state index."""
        obs = np.atleast_1d(obs).astype(float)
        # Bin each dimension into n_state_bins
        state_idx = tuple(np.clip(
            (obs * self.n_state_bins).astype(int),
            0,
            self.n_state_bins - 1
        ))
        return state_idx
    
    def select_action(self, obs: np.ndarray, mask: np.ndarray | None = None) -> int:
        """Epsilon-greedy action selection."""
        state_idx = self._state_to_idx(obs)
        
        if mask is None:
            mask = np.ones(self.n_actions, dtype=bool)
        
        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            valid_actions = np.where(mask)[0]
            return int(np.random.choice(valid_actions))
        else:
            # Greedy: pick best Q-value for valid actions
            q_values = []
            for a in range(self.n_actions):
                if mask[a]:
                    q_val = self.Q.get((state_idx, a), 0.0)
                    q_values.append((a, q_val))
                else:
                    q_values.append((a, -np.inf))
            
            best_action = max(q_values, key=lambda x: x[1])[0]
            return best_action
    
    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Update Q-value using Q-Learning update rule."""
        state_idx = self._state_to_idx(obs)
        next_state_idx = self._state_to_idx(next_obs)
        
        # Get current Q-value
        current_q = self.Q.get((state_idx, action), 0.0)
        
        # Get max Q-value for next state
        if done:
            max_next_q = 0.0
        else:
            max_next_q = max(
                [self.Q.get((next_state_idx, a), 0.0) for a in range(self.n_actions)],
                default=0.0
            )
        
        # Q-Learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.Q[(state_idx, action)] = new_q
        
        self.training_steps += 1
    
    def predict(self, obs: np.ndarray, mask: np.ndarray | None = None) -> int:
        """Greedy action selection (no exploration)."""
        state_idx = self._state_to_idx(obs)
        
        if mask is None:
            mask = np.ones(self.n_actions, dtype=bool)
        
        q_values = []
        for a in range(self.n_actions):
            if mask[a]:
                q_val = self.Q.get((state_idx, a), 0.0)
                q_values.append((a, q_val))
            else:
                q_values.append((a, -np.inf))
        
        best_action = max(q_values, key=lambda x: x[1])[0]
        return best_action
    
    def save(self, path: str | Path):
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str | Path) -> QLearning:
        """Load model from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def get_stats(self) -> dict[str, Any]:
        """Get training statistics."""
        return {
            "n_actions": self.n_actions,
            "n_state_bins": self.n_state_bins,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "n_state_action_pairs": len(self.Q),
        }


class DeepQLearning:
    """Deep Q-Learning with neural network approximation (stub for now)."""
    
    def __init__(self, n_actions: int, d: int):
        """Initialize Deep Q-Learning."""
        raise NotImplementedError("DeepQLearning coming soon - use QLearning for now")

