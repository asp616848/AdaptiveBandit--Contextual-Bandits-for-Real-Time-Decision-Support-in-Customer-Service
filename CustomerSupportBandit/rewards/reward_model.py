"""
Learned Reward Model.

Trained on OpenAssistant human quality ratings to predict conversation
quality from partial trajectories. Used to:
1. Validate escalation risk proxies from Twitter/Reddit
2. Provide shaped intermediate rewards for RL training
3. Calibrate the economic reward function

Architecture: Simple regression model mapping conversation features
to predicted quality score [0, 1].
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.feature_engineer import (
    compute_sentiment_trajectory,
    compute_complexity_features,
    compute_escalation_features,
)


class RewardModel:
    """
    Learned reward model calibrated on OpenAssistant quality ratings.

    Maps conversation features → predicted quality score.
    Uses ridge regression on hand-crafted features for interpretability.
    """

    def __init__(self, feature_dim: int = 15, lambda_reg: float = 1.0):
        """
        Parameters
        ----------
        feature_dim : int
            Number of conversation-level features (subset of full features,
            excluding tier info — quality should be tier-independent).
        lambda_reg : float
            Ridge regression regularization.
        """
        self.feature_dim = feature_dim
        self.lambda_reg = lambda_reg
        self.weights = np.zeros(feature_dim)
        self.bias = 0.5  # Prior: median quality
        self.is_trained = False
        self.training_stats = {}

    def extract_quality_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract features relevant to conversation quality prediction.

        These are tier-independent features (sentiment, complexity, etc.)
        that determine conversation quality.
        """
        sent_feats = compute_sentiment_trajectory(texts)
        complexity_feats = compute_complexity_features(texts)
        esc_feats = compute_escalation_features(texts)

        features = list(sent_feats.values()) + list(complexity_feats.values()) + list(esc_feats.values())
        return np.array(features[:self.feature_dim], dtype=np.float32)

    def predict_quality(self, texts: List[str]) -> float:
        """
        Predict quality score for a conversation.

        Returns
        -------
        float
            Predicted quality in [0, 1].
        """
        features = self.extract_quality_features(texts)
        raw_score = self.weights @ features + self.bias
        # Sigmoid to bound in [0, 1]
        return 1.0 / (1.0 + np.exp(-raw_score))

    def predict_batch(self, text_lists: List[List[str]]) -> np.ndarray:
        """Predict quality for a batch of conversations."""
        features = np.array([self.extract_quality_features(texts)
                            for texts in text_lists])
        raw_scores = features @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-raw_scores))

    def train(self, text_lists: List[List[str]],
              quality_scores: np.ndarray) -> Dict:
        """
        Train the reward model on OpenAssistant quality data.

        Parameters
        ----------
        text_lists : list of list of str
            Conversation texts.
        quality_scores : np.ndarray
            Human-annotated quality scores [0, 1].

        Returns
        -------
        dict
            Training statistics.
        """
        n = len(text_lists)
        assert n == len(quality_scores)

        # Extract features
        X = np.array([self.extract_quality_features(texts)
                      for texts in text_lists])

        # Transform targets to logit space for linear regression
        y = quality_scores.copy()
        y = np.clip(y, 0.01, 0.99)
        y_logit = np.log(y / (1 - y))

        # Ridge regression: w = (X^T X + lambda I)^{-1} X^T y
        XtX = X.T @ X + self.lambda_reg * np.eye(self.feature_dim)
        Xty = X.T @ y_logit

        self.weights = np.linalg.solve(XtX, Xty)
        self.bias = np.mean(y_logit) - self.weights @ np.mean(X, axis=0)

        # Compute training stats
        predictions = self.predict_batch(text_lists)
        mse = np.mean((predictions - quality_scores) ** 2)
        correlation = np.corrcoef(predictions, quality_scores)[0, 1]

        self.is_trained = True
        self.training_stats = {
            'n_samples': n,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'correlation': correlation,
            'weight_norm': np.linalg.norm(self.weights),
        }

        print(f"  Reward model trained on {n:,} samples")
        print(f"    MSE: {mse:.4f} | Correlation: {correlation:.4f}")

        return self.training_stats

    def get_shaped_reward(self, texts_before: List[str],
                           texts_after: List[str]) -> float:
        """
        Compute shaped intermediate reward as quality improvement.

        shaped_reward = Q(s_{t+1}) - Q(s_t)

        This provides potential-based shaping that doesn't alter
        the optimal policy (Ng et al., 1999).
        """
        q_before = self.predict_quality(texts_before)
        q_after = self.predict_quality(texts_after)
        return q_after - q_before

    def get_model_info(self) -> Dict:
        """Return model information."""
        return {
            'is_trained': self.is_trained,
            'feature_dim': self.feature_dim,
            'lambda_reg': self.lambda_reg,
            **self.training_stats,
        }
