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
        self.eta = float(params.get("eta_success", 5.0))
        self.lambda_turn = float(params.get("lambda_turn", 0.15))
        self.omega = float(params.get("omega", 0.16666666666666666))

        self.tier_config = tier_config
        self.kappa = float(tier_config.get("kappa", 0.5))
        self.base_value_by_tier = dict(tier_config.get("base_value_by_tier", {}))

        escalation_costs = reward_params.get("parameters", {}).get("escalation_costs")
        if not escalation_costs:
            escalation_costs = tier_config.get("escalation_cost_by_tier", {})
        self.escalation_costs = {k: float(v) for k, v in escalation_costs.items()}
        self.escalation_bonus_enterprise = float(
            reward_params.get("parameters", {}).get("escalation_bonus_enterprise", 1.0)
        )

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

    def per_turn_reward(self) -> float:
        return float(-self.lambda_turn)

    def terminal_reward(self, outcome: str, state: dict[str, Any], tier: str, value_weight: float) -> float:
        frustration = float(state.get("frustration", 0.0))
        failed_streak = int(state.get("failed_streak", 0))
        turn_count = int(state.get("turn_count", 0))
        tau = float(state.get("tau", 0.5))

        if outcome == "success":
            p_churn_terminal = 0.0
        elif outcome == "dropout":
            p_churn_terminal = 1.0
        elif outcome == "escalation":
            # Bot escalating to human reduces churn more than bot timing out:
            # human agents salvage frustrated/stuck customers far better than a bot giving up.
            # churn_saved scales with how "stuck" the situation is:
            #   frustration=0.9 + streak=3 → saves 70% of potential churn
            #   frustration=0.1 + streak=0 → saves ~6% (unnecessary escalation, barely helps)
            p_churn_base = self.compute_p_churn(frustration, failed_streak, turn_count, tau)
            churn_saved = min(frustration * 0.6 + failed_streak * 0.1, 0.7)
            p_churn_terminal = p_churn_base * (1.0 - churn_saved)
        elif outcome == "timeout":
            p_churn_terminal = self.compute_p_churn(frustration, failed_streak, turn_count, tau)
        else:
            p_churn_terminal = self.compute_p_churn(frustration, failed_streak, turn_count, tau)

        value_at_risk = self.compute_V(tier=tier, value_weight=float(value_weight))
        reward = -self.omega * p_churn_terminal * value_at_risk

        if outcome == "success":
            reward += self.eta
        elif outcome == "escalation":
            # Explicit escalation cost, reduced when context warrants it.
            # appropriateness cancels the cost entirely when bot is clearly failing:
            #   frustration≥0.9 + streak≥3 (Free): appropriateness≥4.8 > cost=4.0 → cost=0
            #   frustration≥0.7 + streak≥2 (Pro):  appropriateness≥3.5 > cost=2.0 → cost=0
            base_cost = float(self.escalation_costs.get(tier, 0.0))
            appropriateness = frustration * 3.0 + min(failed_streak * 0.7, 2.5)
            effective_cost = max(base_cost - appropriateness, 0.0)
            reward -= effective_cost
            if tier == "Enterprise":
                reward += self.escalation_bonus_enterprise
        elif outcome == "unresolved_close":
            reward -= 1.0

        return float(np.clip(reward, -5.0, 5.0))
