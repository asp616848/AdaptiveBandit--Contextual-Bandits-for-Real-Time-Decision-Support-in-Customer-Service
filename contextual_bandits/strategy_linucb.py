"""Strategy-based Contextual Bandit for SupportEnv.

Instead of deciding actions per-turn, select ONE strategy for the entire conversation.
This is a true single-step CB problem suitable for LinUCB.

Strategies:
    0: Escalate Fast - Escalate immediately (or at turn 3 due to masking)
    1: Solve Patiently - Ask questions, provide solutions, never escalate
    2: Adaptive Repair - Use a pure heuristic based on the current conversation state
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any
import pickle
from pathlib import Path


STRATEGIES = {
    0: "escalate_fast",
    1: "solve_patiently",
    2: "adaptive_repair",
}

STRATEGY_DESCRIPTIONS = {
    0: "Escalate at first opportunity (turn 3+ due to masking)",
    1: "Ask info, provide solutions, avoid escalation",
    2: "Use a heuristic repair flow driven by the current observation",
}


@dataclass
class StrategyContext:
    """Context for strategy selection."""
    query_embedding: np.ndarray  # 9D observation
    tier: str  # Free, Pro, Business, Enterprise
    persona: str  # Customer personality type
    
    def to_features(self) -> np.ndarray:
        """Convert context to feature vector for LinUCB."""
        # Use observation as features (9D)
        return self.query_embedding.astype(float)


class StrategyExecutor:
    """Execute a fixed strategy throughout a conversation."""
    
    def __init__(self, strategy: int):
        """
        Initialize executor for a strategy.
        
        Args:
            strategy: Strategy ID (0: escalate_fast, 1: solve_patiently, 2: adaptive_repair)
        """
        self.strategy = strategy
        self.strategy_name = STRATEGIES[strategy]
    
    def get_action(
        self,
        obs: np.ndarray,
        turn: int,
    ) -> int:
        """
        Get action for current turn based on strategy.
        
        Args:
            obs: Current observation
            turn: Current turn number
        Returns:
            Action to take
        """
        if self.strategy == 0:  # escalate_fast
            # Try to escalate as soon as allowed (turn 3+)
            if turn >= 3:
                return 3  # Escalate
            else:
                return 0  # AskInfo until escalation is allowed
        
        elif self.strategy == 1:  # solve_patiently
            # Ask info early, provide solution later, never escalate
            if turn < 2:
                return 0  # AskInfo
            elif turn < 4:
                return 1  # ProvideSolution
            else:
                return 4  # Close (don't escalate)
        
        elif self.strategy == 2:  # adaptive_repair
            # Pure heuristic: adapt to the observed state without any PPO oracle.
            sentiment = float(obs[2]) if len(obs) > 2 else 0.0
            confidence = float(obs[1]) if len(obs) > 1 else 0.0
            escalation_flag = float(obs[4]) if len(obs) > 4 else 0.0
            turn_norm = float(obs[6]) if len(obs) > 6 else 0.0

            if escalation_flag > 0.5 or sentiment < -0.3:
                return 2  # AffectiveRepair
            if turn >= 3 and confidence < 0.5:
                return 3  # Escalate
            if turn_norm > 0.6:
                return 4  # Close
            if confidence < 0.4:
                return 0  # AskInfo
            return 1  # ProvideSolution
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


class StrategyLinUCB:
    """LinUCB for selecting strategies instead of per-turn actions."""
    
    def __init__(
        self,
        n_strategies: int = 3,
        d: int = 9,
        alpha: float = 1.0,
    ):
        """
        Initialize Strategy LinUCB.
        
        Args:
            n_strategies: Number of strategies (default 3)
            d: Feature dimension (observation size)
            alpha: Exploration parameter
        """
        self.n_strategies = n_strategies
        self.d = d
        self.alpha = alpha
        
        # One model per strategy
        self.models = {}
        for s in range(n_strategies):
            self.models[s] = {
                "A": np.eye(d),  # Gram matrix
                "b": np.zeros(d),  # Reward vector
            }
        
        # Statistics
        self.strategy_counts = np.zeros(n_strategies)
        self.strategy_rewards = [[] for _ in range(n_strategies)]
    
    def select_strategy(self, context: np.ndarray, mask: np.ndarray | None = None) -> int:
        """
        Select strategy using UCB.
        
        Args:
            context: Feature vector (d,)
            mask: Strategy mask (n_strategies,) where True = valid
        
        Returns:
            Selected strategy
        """
        context = np.atleast_1d(context).astype(float)
        
        if mask is None:
            mask = np.ones(self.n_strategies, dtype=bool)
        
        # Compute UCB for each strategy
        ucbs = []
        for s in range(self.n_strategies):
            if not mask[s]:
                ucbs.append(-np.inf)
                continue
            
            A = self.models[s]["A"]
            b = self.models[s]["b"]
            
            try:
                theta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                ucbs.append(-np.inf)
                continue
            
            # Mean: theta^T x
            mu = theta @ context
            
            # Confidence: alpha * sqrt(x^T A^{-1} x)
            try:
                A_inv = np.linalg.inv(A)
                confidence = self.alpha * np.sqrt(context @ A_inv @ context)
            except np.linalg.LinAlgError:
                confidence = self.alpha
            
            ucb = mu + confidence
            ucbs.append(ucb)
        
        strategy = int(np.argmax(ucbs))
        return strategy
    
    def update(self, context: np.ndarray, strategy: int, reward: float):
        """
        Update model with observed reward for strategy.
        
        Args:
            context: Feature vector (d,)
            strategy: Selected strategy
            reward: Observed reward (e.g., 1 if resolved, 0 if not)
        """
        context = np.atleast_1d(context).astype(float)
        reward = float(reward)
        
        # Update sufficient statistics for selected strategy
        self.models[strategy]["A"] += np.outer(context, context)
        self.models[strategy]["b"] += reward * context
        
        self.strategy_counts[strategy] += 1
        self.strategy_rewards[strategy].append(reward)
    
    def predict(self, context: np.ndarray, mask: np.ndarray | None = None) -> int:
        """Predict best strategy (exploitation only)."""
        context = np.atleast_1d(context).astype(float)
        
        if mask is None:
            mask = np.ones(self.n_strategies, dtype=bool)
        
        means = []
        for s in range(self.n_strategies):
            if not mask[s]:
                means.append(-np.inf)
                continue
            
            try:
                theta = np.linalg.solve(self.models[s]["A"], self.models[s]["b"])
                mu = theta @ context
                means.append(mu)
            except np.linalg.LinAlgError:
                means.append(-np.inf)
        
        return int(np.argmax(means))
    
    def save(self, path: str | Path):
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str | Path) -> StrategyLinUCB:
        """Load model from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def get_stats(self) -> dict[str, Any]:
        """Get training statistics."""
        return {
            "n_strategies": self.n_strategies,
            "d": self.d,
            "alpha": self.alpha,
            "strategy_counts": self.strategy_counts.tolist(),
            "mean_rewards_per_strategy": [
                np.mean(rewards) if rewards else 0.0
                for rewards in self.strategy_rewards
            ],
        }
