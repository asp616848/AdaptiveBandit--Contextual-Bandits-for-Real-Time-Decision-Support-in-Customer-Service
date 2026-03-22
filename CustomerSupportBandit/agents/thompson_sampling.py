"""
Thompson Sampling Contextual Bandit Agent.

Implements Bayesian linear regression with Thompson Sampling for
customer support routing decisions.

Maintains a posterior distribution over action-value functions and
samples from it to balance exploration and exploitation.
"""

import numpy as np
from typing import Dict

from .base_agent import BaseAgent


class ThompsonSamplingAgent(BaseAgent):
    """
    Thompson Sampling with Bayesian Linear Regression.

    For each action a, maintains:
    - B_a: (d x d) precision matrix (inverse of posterior covariance)
    - mu_a: (d,) posterior mean
    - f_a: (d,) accumulated reward-weighted features

    At each round, samples theta_a ~ N(mu_a, v^2 * B_a^{-1})
    and selects the action with highest sampled value theta_a^T x.
    """

    def __init__(self, n_actions: int = 2, feature_dim: int = 23,
                 v_squared: float = 1.0, lambda_prior: float = 1.0):
        """
        Parameters
        ----------
        n_actions : int
            Number of actions.
        feature_dim : int
            Context dimension.
        v_squared : float
            Variance scaling parameter for posterior sampling.
            Controls exploration intensity.
        lambda_prior : float
            Prior precision (regularization).
        """
        super().__init__(n_actions, feature_dim, name="ThompsonSampling")
        self.v_squared = v_squared
        self.lambda_prior = lambda_prior

        # Per-action Bayesian parameters
        self.B = [lambda_prior * np.eye(feature_dim) for _ in range(n_actions)]
        self.f = [np.zeros(feature_dim) for _ in range(n_actions)]
        self.B_inv = [(1.0/lambda_prior) * np.eye(feature_dim) for _ in range(n_actions)]
        self.mu = [np.zeros(feature_dim) for _ in range(n_actions)]

    def select_action(self, context: np.ndarray) -> int:
        """
        Select action via Thompson Sampling.

        Sample theta_a from posterior and pick action with max theta_a^T x.
        """
        x = context.reshape(-1)
        assert len(x) == self.feature_dim

        sampled_values = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            # Sample from posterior: theta ~ N(mu, v^2 * B^{-1})
            try:
                theta_sample = np.random.multivariate_normal(
                    self.mu[a],
                    self.v_squared * self.B_inv[a]
                )
            except np.linalg.LinAlgError:
                # Fallback: diagonal sampling
                var = self.v_squared * np.diag(self.B_inv[a])
                theta_sample = np.random.normal(self.mu[a], np.sqrt(np.abs(var)))

            sampled_values[a] = theta_sample @ x

        action = int(np.argmax(sampled_values))

        self.t += 1
        self.action_counts[action] += 1
        return action

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        """
        Update posterior for the chosen action.

        B_a += x x^T
        f_a += reward * x
        mu_a = B_a^{-1} f_a
        """
        x = context.reshape(-1)

        # Update precision matrix and feature accumulator
        self.B[action] += np.outer(x, x)
        self.f[action] += reward * x

        # Sherman-Morrison for inverse update
        x_col = x.reshape(-1, 1)
        B_inv = self.B_inv[action]
        numerator = B_inv @ x_col @ x_col.T @ B_inv
        denominator = 1.0 + (x_col.T @ B_inv @ x_col).item()
        self.B_inv[action] = B_inv - numerator / denominator

        # Update posterior mean
        self.mu[action] = self.B_inv[action] @ self.f[action]

        self.total_reward += reward
        self.reward_history.append(reward)

    def get_posterior_info(self) -> Dict:
        """Return posterior statistics for each action."""
        info = {}
        for a in range(self.n_actions):
            posterior_var = np.diag(self.B_inv[a])
            info[a] = {
                'mean_weights': self.mu[a],
                'posterior_uncertainty': np.sqrt(np.mean(posterior_var)),
                'weight_norm': np.linalg.norm(self.mu[a]),
            }
        return info

    def get_policy_info(self) -> Dict:
        info = super().get_policy_info()
        info['v_squared'] = self.v_squared
        posterior = self.get_posterior_info()
        info['posterior_uncertainty'] = {
            a: p['posterior_uncertainty'] for a, p in posterior.items()
        }
        return info

    def reset(self) -> None:
        super().reset()
        self.B = [self.lambda_prior * np.eye(self.feature_dim)
                  for _ in range(self.n_actions)]
        self.f = [np.zeros(self.feature_dim) for _ in range(self.n_actions)]
        self.B_inv = [(1.0/self.lambda_prior) * np.eye(self.feature_dim)
                      for _ in range(self.n_actions)]
        self.mu = [np.zeros(self.feature_dim) for _ in range(self.n_actions)]
