"""
Abstract base class for all customer support routing agents.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple


class BaseAgent(ABC):
    """
    Base class for customer support routing agents.

    All agents implement:
    - select_action(context): choose action given context features
    - update(context, action, reward): learn from feedback
    - get_policy_info(): return interpretable policy state
    """

    def __init__(self, n_actions: int, feature_dim: int, name: str = "BaseAgent"):
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.name = name
        self.t = 0  # timestep counter
        self.total_reward = 0.0
        self.action_counts = np.zeros(n_actions)
        self.reward_history = []

    @abstractmethod
    def select_action(self, context: np.ndarray) -> int:
        """
        Select an action given the current context.

        Parameters
        ----------
        context : np.ndarray
            Feature vector of shape (feature_dim,).

        Returns
        -------
        int
            Selected action index.
        """
        pass

    @abstractmethod
    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        """
        Update the agent's model given observed feedback.

        Parameters
        ----------
        context : np.ndarray
            Feature vector used for decision.
        action : int
            Action that was taken.
        reward : float
            Observed reward.
        """
        pass

    def get_policy_info(self) -> Dict:
        """Return interpretable information about the current policy."""
        return {
            'name': self.name,
            'timestep': self.t,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(self.t, 1),
            'action_distribution': self.action_counts / max(self.action_counts.sum(), 1),
        }

    def reset(self) -> None:
        """Reset the agent to its initial state."""
        self.t = 0
        self.total_reward = 0.0
        self.action_counts = np.zeros(self.n_actions)
        self.reward_history = []
