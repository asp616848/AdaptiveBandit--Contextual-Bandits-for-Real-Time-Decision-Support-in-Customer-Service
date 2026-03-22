"""
Rule-based baseline agent.

Implements simple tier-based routing rules as described in the proposal:
"A naive rule (route all Business+ to humans, all Free to bots) ignores
conversation-level signal."

This baseline is included to quantify the marginal value of the ML investment.
"""

import numpy as np
from typing import Dict

from .base_agent import BaseAgent

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import TIER_CONFIG, TIER_NAMES


class RuleBasedAgent(BaseAgent):
    """
    Rule-based escalation policy.

    Rules:
    - Enterprise/Business+: escalate if frustration score > threshold
      OR escalation phrases detected
    - Pro: escalate only if escalation phrases detected AND sentiment negative
    - Free: never escalate (always bot)

    Feature indices (from feature_engineer.py):
    - sentiment_current: 0
    - has_escalation_phrase: 13
    - frustration_score: 14
    - tier_idx: 15
    """

    def __init__(self, n_actions: int = 2, feature_dim: int = 23,
                 frustration_threshold: float = 2.0,
                 sentiment_threshold: float = -0.3):
        super().__init__(n_actions, feature_dim, name="RuleBasedAgent")
        self.frustration_threshold = frustration_threshold
        self.sentiment_threshold = sentiment_threshold

    def select_action(self, context: np.ndarray) -> int:
        """
        Select action based on hard-coded rules.

        Actions: 0 = bot, 1 = human
        """
        tier_idx = int(context[15])
        sentiment = context[0]
        has_escalation = context[13]
        frustration = context[14]

        # Free tier: always bot
        if tier_idx == 0:
            action = 0

        # Pro tier: escalate only on phrases + negative sentiment
        elif tier_idx == 1:
            if has_escalation > 0 and sentiment < self.sentiment_threshold:
                action = 1
            else:
                action = 0

        # Business+ or Enterprise: more aggressive escalation
        else:
            if (frustration > self.frustration_threshold or
                has_escalation > 0 or
                sentiment < self.sentiment_threshold):
                action = 1
            else:
                action = 0

        self.t += 1
        self.action_counts[action] += 1
        return action

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        """Rule-based agent does not learn."""
        self.total_reward += reward
        self.reward_history.append(reward)

    def get_policy_info(self) -> Dict:
        info = super().get_policy_info()
        info['frustration_threshold'] = self.frustration_threshold
        info['sentiment_threshold'] = self.sentiment_threshold
        return info
