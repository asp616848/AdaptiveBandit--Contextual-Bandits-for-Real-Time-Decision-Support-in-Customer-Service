from __future__ import annotations

import math
from typing import Any

import numpy as np


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-x)))


class RewardEngine:
    """Reward model with terminal-only churn loss."""

    def __init__(self, reward_params: dict[str, Any], tier_config: dict[str, Any]):
        params = reward_params.get("parameters", {})
        self.eta = float(params.get("eta_success", 12.0))
        self.lambda_turn = float(params.get("lambda_turn", 0.05))
        self.omega = float(params.get("omega", 0.16666666666666666))
        self.reward_min = float(params.get("reward_min", -10.0))
        self.reward_max = float(params.get("reward_max", 10.0))

        self.tier_config = tier_config
        self.kappa = float(tier_config.get("kappa", 0.5))
        self.base_value_by_tier = dict(tier_config.get("base_value_by_tier", {}))

        escalation_costs = reward_params.get("parameters", {}).get("escalation_costs")
        if not escalation_costs:
            escalation_costs = tier_config.get("escalation_cost_by_tier", {})
        self.escalation_costs = {k: float(v) for k, v in escalation_costs.items()}
        self.escalation_bonus_enterprise = float(
            reward_params.get("parameters", {}).get("escalation_bonus_enterprise", 0.0)
        )
        self.escalation_penalty = float(params.get("escalation_penalty", 6.0))
        self.dropout_penalty = float(params.get("dropout_penalty", 10.0))
        self.unresolved_close_penalty = float(params.get("unresolved_close_penalty", 8.0))
        self.frustration_penalty = float(params.get("frustration_penalty", 1.0))
        self.engagement_bonus = float(params.get("engagement_bonus", 0.5))
        self.sentiment_improvement_bonus = float(params.get("sentiment_improvement_bonus", 2.0))

        churn_coeffs = reward_params.get("churn_model", {}).get("coefficients")
        if not churn_coeffs:
            churn_coeffs = {"c0": -3.8, "cf": 3.0, "cs": 0.1, "ct": 0.08, "ctau": -1.5}
        self.churn_coeffs = {k: float(v) for k, v in churn_coeffs.items()}

    def compute_V(self, tier: str, value_weight: float) -> float:
        v_base = float(self.base_value_by_tier.get(tier, 1.0))
        return float(v_base * (1.0 + self.kappa * float(value_weight)))

    def compute_p_churn(self, frustration: float, failed_streak: int, turn_count: int, tau: float) -> float:
        c = self.churn_coeffs
        z = c["c0"] + c["cf"] * frustration + c["cs"] * failed_streak + c["ct"] * turn_count + c["ctau"] * tau
        return sigmoid(z)

    def per_turn_reward(self, state: dict[str, Any] | None = None) -> float:
        penalty = self.lambda_turn
        if state is not None:
            penalty += float(state.get('frustration', 0.0)) * self.frustration_penalty
        return float(-penalty)

    def terminal_reward(self, outcome: str, state: dict[str, Any], tier: str, value_weight: float) -> float:
        if outcome == "success":
            p_churn_terminal = 0.0
        elif outcome == "dropout":
            p_churn_terminal = 1.0
        elif outcome in {"escalation", "timeout"}:
            p_churn_terminal = self.compute_p_churn(
                frustration=float(state["frustration"]),
                failed_streak=int(state["failed_streak"]),
                turn_count=int(state["turn_count"]),
                tau=float(state["tau"]),
            )
        else:
            p_churn_terminal = self.compute_p_churn(
                frustration=float(state["frustration"]),
                failed_streak=int(state["failed_streak"]),
                turn_count=int(state["turn_count"]),
                tau=float(state["tau"]),
            )

        value_at_risk = self.compute_V(tier=tier, value_weight=float(value_weight))
        reward = -self.omega * p_churn_terminal * value_at_risk

        if outcome == "success":
            reward += self.eta
        elif outcome == "escalation":
            reward -= self.escalation_penalty
            reward -= float(self.escalation_costs.get(tier, 0.0))
            if tier == "Enterprise":
                reward += self.escalation_bonus_enterprise
        elif outcome == "dropout":
            reward -= self.dropout_penalty
        elif outcome == "unresolved_close":
            reward -= self.unresolved_close_penalty

        return float(np.clip(reward, self.reward_min, self.reward_max))
