"""Linear Upper Confidence Bound (LinUCB) contextual bandit algorithm.

Implements the LinUCB algorithm from:
  "A Contextual-Bandit Approach to Personalized News Recommendation" (Li et al., 2010)

Also includes PerTurnLinUCBPolicy which trains separate LinUCB models for each turn.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any
import pickle
from pathlib import Path


@dataclass
class LinUCBArm:
    """Linear regression model for a single arm (action)."""
    d: int  # Feature dimension
    alpha: float = 1.0  # Exploration parameter
    
    # Sufficient statistics
    A: np.ndarray = field(default_factory=lambda: None)
    b: np.ndarray = field(default_factory=lambda: None)
    
    def __post_init__(self):
        if self.A is None:
            self.A = np.eye(self.d)
        if self.b is None:
            self.b = np.zeros(self.d)
    
    def update(self, x: np.ndarray, reward: float):
        """Update arm model with new (context, reward) pair."""
        x = np.atleast_1d(x).astype(float)
        self.A += np.outer(x, x)
        self.b += reward * x
    
    def predict_ucb(self, x: np.ndarray) -> tuple[float, float]:
        """
        Predict reward and confidence bound for context x.
        
        Returns:
            (predicted_reward, upper_confidence_bound)
        """
        x = np.atleast_1d(x).astype(float)
        
        # Solve A * theta = b
        try:
            theta = np.linalg.solve(self.A, self.b)
        except np.linalg.LinAlgError:
            # Singular matrix - return conservative UCB
            return 0.0, float('inf')
        
        # Predicted reward: theta^T x
        mu = theta @ x
        
        # Confidence radius: alpha * sqrt(x^T A^{-1} x)
        try:
            A_inv = np.linalg.inv(self.A)
            confidence = self.alpha * np.sqrt(x @ A_inv @ x)
        except np.linalg.LinAlgError:
            confidence = self.alpha  # Fallback
        
        ucb = mu + confidence
        return mu, ucb


class LinUCB:
    """Linear Contextual Bandit Algorithm (LinUCB)."""
    
    def __init__(
        self,
        n_actions: int,
        d: int,
        alpha: float = 1.0,
        name: str = "LinUCB",
    ):
        """
        Initialize LinUCB policy.
        
        Args:
            n_actions: Number of actions
            d: Feature dimension
            alpha: Exploration parameter (higher = more exploration)
            name: Policy name
        """
        self.n_actions = n_actions
        self.d = d
        self.alpha = alpha
        self.name = name
        
        # Initialize arm models
        self.arms = [LinUCBArm(d=d, alpha=alpha) for _ in range(n_actions)]
        
        # Statistics
        self.t = 0  # Total time steps
        self.action_counts = np.zeros(n_actions)
        self.rewards_per_action = [[] for _ in range(n_actions)]
    
    def select_action(self, x: np.ndarray, mask: np.ndarray | None = None) -> int:
        """
        Select action using UCB exploration.
        
        Args:
            x: Context vector (d,)
            mask: Action mask (n_actions,) where True = valid action
        
        Returns:
            Selected action index
        """
        x = np.atleast_1d(x).astype(float)
        
        if mask is None:
            mask = np.ones(self.n_actions, dtype=bool)
        
        # Compute UCB for all valid actions
        ucbs = []
        for i, arm in enumerate(self.arms):
            if mask[i]:
                _, ucb = arm.predict_ucb(x)
                ucbs.append(ucb)
            else:
                ucbs.append(-np.inf)  # Block invalid actions
        
        # Select action with highest UCB
        action = int(np.argmax(ucbs))
        return action
    
    def update(self, x: np.ndarray, action: int, reward: float):
        """
        Update model with observed (context, action, reward).
        
        Args:
            x: Context vector (d,)
            action: Selected action
            reward: Observed reward
        """
        x = np.atleast_1d(x).astype(float)
        reward = float(reward)
        
        self.arms[action].update(x, reward)
        self.action_counts[action] += 1
        self.rewards_per_action[action].append(reward)
        self.t += 1
    
    def predict(self, x: np.ndarray, mask: np.ndarray | None = None) -> int:
        """Predict best action (exploitation, no exploration)."""
        x = np.atleast_1d(x).astype(float)
        
        if mask is None:
            mask = np.ones(self.n_actions, dtype=bool)
        
        # Compute mean reward for all valid actions
        means = []
        for i, arm in enumerate(self.arms):
            if mask[i]:
                try:
                    theta = np.linalg.solve(arm.A, arm.b)
                    mu = theta @ x
                    means.append(mu)
                except np.linalg.LinAlgError:
                    means.append(-np.inf)
            else:
                means.append(-np.inf)
        
        return int(np.argmax(means))
    
    def save(self, path: str | Path):
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str | Path) -> LinUCB:
        """Load model from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def get_stats(self) -> dict[str, Any]:
        """Get training statistics."""
        return {
            "n_actions": self.n_actions,
            "d": self.d,
            "alpha": self.alpha,
            "total_timesteps": self.t,
            "action_counts": self.action_counts.tolist(),
            "mean_rewards_per_action": [
                np.mean(rewards) if rewards else 0.0
                for rewards in self.rewards_per_action
            ],
        }


class PerTurnLinUCBPolicy:
    """Per-turn contextual bandit: separate LinUCB model for each turn."""
    
    def __init__(
        self,
        n_turns: int = 20,
        n_actions: int = 5,
        d: int = 9,
        alpha: float = 1.0,
    ):
        """
        Initialize per-turn LinUCB policy.
        
        Args:
            n_turns: Maximum number of turns
            n_actions: Number of actions
            d: Feature dimension (observation size)
            alpha: Exploration parameter
        """
        self.n_turns = n_turns
        self.n_actions = n_actions
        self.d = d
        self.alpha = alpha
        
        # One LinUCB model per turn
        self.models = {
            turn: LinUCB(n_actions, d, alpha, name=f"LinUCB-Turn{turn}")
            for turn in range(n_turns)
        }
    
    def select_action(
        self,
        obs: np.ndarray,
        turn: int,
        mask: np.ndarray | None = None,
    ) -> int:
        """Select action for given turn using per-turn LinUCB."""
        if turn not in self.models:
            # If turn exceeds max, use last model
            turn = self.n_turns - 1
        
        return self.models[turn].select_action(obs, mask)
    
    def predict_action(
        self,
        obs: np.ndarray,
        turn: int,
        mask: np.ndarray | None = None,
    ) -> int:
        """Predict best action (exploitation only)."""
        if turn not in self.models:
            turn = self.n_turns - 1
        
        return self.models[turn].predict(obs, mask)
    
    def update(
        self,
        obs: np.ndarray,
        turn: int,
        action: int,
        reward: float,
    ):
        """Update model for given turn."""
        if turn not in self.models:
            turn = self.n_turns - 1
        
        self.models[turn].update(obs, action, reward)
    
    def save(self, path: str | Path):
        """Save all models to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        for turn, model in self.models.items():
            model.save(path / f"linucb_turn_{turn}.pkl")
    
    @staticmethod
    def load(path: str | Path) -> PerTurnLinUCBPolicy:
        """Load models from directory."""
        path = Path(path)
        
        # Find all turn models
        turn_files = sorted(path.glob("linucb_turn_*.pkl"))
        if not turn_files:
            raise FileNotFoundError(f"No LinUCB models found in {path}")
        
        # Load first model to get config
        first_model = LinUCB.load(turn_files[0])
        policy = PerTurnLinUCBPolicy(
            n_turns=len(turn_files),
            n_actions=first_model.n_actions,
            d=first_model.d,
            alpha=first_model.alpha,
        )
        
        # Load all models
        for turn_file in turn_files:
            turn = int(turn_file.stem.split('_')[-1])
            policy.models[turn] = LinUCB.load(turn_file)
        
        return policy
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics for all turns."""
        return {
            turn: model.get_stats()
            for turn, model in self.models.items()
        }
