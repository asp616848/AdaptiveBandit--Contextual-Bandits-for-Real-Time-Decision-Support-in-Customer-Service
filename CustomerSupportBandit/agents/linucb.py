"""
LinUCB Contextual Bandit Agent.

Implements the LinUCB algorithm (Li et al., 2010) for customer support routing.
Each action maintains a separate linear model with confidence bounds.

Reference: "A Contextual-Bandit Approach to Personalized News Article
Recommendation" — Li, Chu, Langford, Schapire (WWW 2010)
"""

import numpy as np
from typing import Dict

from .base_agent import BaseAgent


class LinUCBAgent(BaseAgent):
    """
    LinUCB with disjoint linear models.

    For each action a, maintains:
    - A_a: (d x d) matrix = D_a^T D_a + I  (design matrix)
    - b_a: (d,) vector = D_a^T c_a  (reward-weighted features)

    At each round:
    - theta_a = A_a^{-1} b_a  (ridge regression estimate)
    - p_a = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)  (UCB)
    - Select action with highest UCB.
    """

    def __init__(self, n_actions: int = 2, feature_dim: int = 23,
                 alpha: float = 1.0):
        """
        Parameters
        ----------
        n_actions : int
            Number of actions (2 for bandit: bot/human).
        feature_dim : int
            Dimension of context features.
        alpha : float
            Exploration parameter controlling the width of confidence bounds.
            Higher alpha = more exploration.
        """
        super().__init__(n_actions, feature_dim, name="LinUCB")
        self.alpha = alpha

        # Initialize per-action parameters
        self.A = [np.eye(feature_dim) for _ in range(n_actions)]
        self.b = [np.zeros(feature_dim) for _ in range(n_actions)]
        self.A_inv = [np.eye(feature_dim) for _ in range(n_actions)]

    def select_action(self, context: np.ndarray) -> int:
        """
        Select action using Upper Confidence Bound.

        Parameters
        ----------
        context : np.ndarray
            Feature vector of shape (feature_dim,).

        Returns
        -------
        int
            Action with highest UCB score.
        """
        x = context.reshape(-1)
        assert len(x) == self.feature_dim, \
            f"Expected {self.feature_dim} features, got {len(x)}"

        ucb_scores = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            theta_a = self.A_inv[a] @ self.b[a]
            exploitation = theta_a @ x
            exploration = self.alpha * np.sqrt(x @ self.A_inv[a] @ x)
            ucb_scores[a] = exploitation + exploration

        action = int(np.argmax(ucb_scores))

        self.t += 1
        self.action_counts[action] += 1
        return action

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        """
        Update the linear model for the chosen action.

        Parameters
        ----------
        context : np.ndarray
            Feature vector.
        action : int
            Action taken.
        reward : float
            Observed reward.
        """
        x = context.reshape(-1)

        # Rank-1 update: A_a += x x^T
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x

        # Update inverse using Sherman-Morrison formula for efficiency
        x_col = x.reshape(-1, 1)
        A_inv = self.A_inv[action]
        numerator = A_inv @ x_col @ x_col.T @ A_inv
        denominator = 1.0 + (x_col.T @ A_inv @ x_col).item()
        self.A_inv[action] = A_inv - numerator / denominator

        self.total_reward += reward
        self.reward_history.append(reward)

    def get_action_weights(self) -> Dict[int, np.ndarray]:
        """Return learned weight vectors for each action."""
        weights = {}
        for a in range(self.n_actions):
            weights[a] = self.A_inv[a] @ self.b[a]
        return weights

    def get_policy_info(self) -> Dict:
        info = super().get_policy_info()
        info['alpha'] = self.alpha
        weights = self.get_action_weights()
        info['weight_norms'] = {a: np.linalg.norm(w) for a, w in weights.items()}
        return info

    def reset(self) -> None:
        super().reset()
        self.A = [np.eye(self.feature_dim) for _ in range(self.n_actions)]
        self.b = [np.zeros(self.feature_dim) for _ in range(self.n_actions)]
        self.A_inv = [np.eye(self.feature_dim) for _ in range(self.n_actions)]
