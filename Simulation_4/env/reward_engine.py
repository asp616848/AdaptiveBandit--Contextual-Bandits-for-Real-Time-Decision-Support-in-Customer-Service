from __future__ import annotations

import math
from typing import Any

import numpy as np


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-x)))


class RewardEngine:
    """Reward model supporting dense shaping and legacy terminal churn reward."""

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

        dense_cfg = reward_params.get("dense_reward", {})
        self.use_dense_per_turn = bool(dense_cfg.get("enabled", True))
        self.dense_weights = {
            # Updated shaping weights (stronger gradient toward success)
            "delta_sentiment": float(dense_cfg.get("w_delta_sentiment", 0.5)),
            "delta_frustration": float(dense_cfg.get("w_delta_frustration", -0.2)),
            "delta_progress": float(dense_cfg.get("w_delta_progress", 1.5)),
            "delta_info": float(dense_cfg.get("w_delta_info", 0.8)),
            # Reduce per-turn penalty to avoid overly negative signals
            "step_penalty": float(dense_cfg.get("step_penalty", -0.02)),
        }
        self.terminal_bonus = {
            # Strong positive terminal reward for resolved cases to create clear learning signal
            "success": float(dense_cfg.get("terminal_success", 10.0)),
            "escalation": float(dense_cfg.get("terminal_escalation", -1.5)),
            "dropout": float(dense_cfg.get("terminal_dropout", -1.2)),
            "timeout": float(dense_cfg.get("terminal_timeout", -0.8)),
            "unresolved_close": float(dense_cfg.get("terminal_unresolved_close", -0.8)),
        }

    def compute_V(self, tier: str, value_weight: float) -> float:
        v_base = float(self.base_value_by_tier.get(tier, 1.0))
        return float(v_base * (1.0 + self.kappa * float(value_weight)))

    def compute_p_churn(self, frustration: float, failed_streak: int, turn_count: int, tau: float) -> float:
        c = self.churn_coeffs
        z = c["c0"] + c["cf"] * frustration + c["cs"] * failed_streak + c["ct"] * turn_count + c["ctau"] * tau
        return sigmoid(z)

    def _state_deltas(self, pre_state: dict[str, Any], post_state: dict[str, Any]) -> dict[str, float]:
        delta_frustration = float(post_state.get("frustration", 0.0)) - float(pre_state.get("frustration", 0.0))
        delta_progress = float(post_state.get("progress", 0.0)) - float(pre_state.get("progress", 0.0))
        delta_info = float(post_state.get("information", 0.0)) - float(pre_state.get("information", 0.0))

        # Sentiment proxy is inverse frustration if sentiment is not explicit in state.
        pre_sentiment = float(pre_state.get("sentiment", 1.0 - float(pre_state.get("frustration", 0.0))))
        post_sentiment = float(post_state.get("sentiment", 1.0 - float(post_state.get("frustration", 0.0))))
        delta_sentiment = post_sentiment - pre_sentiment

        return {
            "delta_sentiment": float(delta_sentiment),
            "delta_frustration": float(delta_frustration),
            "delta_progress": float(delta_progress),
            "delta_info": float(delta_info),
        }

    def compute_per_turn_components(self, pre_state: dict[str, Any], post_state: dict[str, Any]) -> dict[str, float]:
        deltas = self._state_deltas(pre_state, post_state)
        if not self.use_dense_per_turn:
            return {
                **deltas,
                "per_turn_reward": float(-self.lambda_turn),
            }

        # Compose per-turn reward with stronger positive components for progress/info/sentiment
        reward = (
            self.dense_weights["delta_sentiment"] * deltas["delta_sentiment"]
            + self.dense_weights["delta_frustration"] * deltas["delta_frustration"]
            + self.dense_weights["delta_progress"] * deltas["delta_progress"]
            + self.dense_weights["delta_info"] * deltas["delta_info"]
            + self.dense_weights["step_penalty"]
        )
        return {
            **deltas,
            "per_turn_reward": float(reward),
        }

    def per_turn_reward(self) -> float:
        return float(-self.lambda_turn)

    def terminal_reward(self, outcome: str, state: dict[str, Any], tier: str, value_weight: float) -> float:
        if self.use_dense_per_turn:
            # Allow larger terminal bonuses so strong successes are not clipped too aggressively
            return float(np.clip(self.terminal_bonus.get(outcome, self.terminal_bonus.get("timeout", -0.8)), -20.0, 20.0))

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
            reward -= float(self.escalation_costs.get(tier, 0.0))
            if tier == "Enterprise":
                reward += self.escalation_bonus_enterprise
        elif outcome == "unresolved_close":
            reward -= 1.0

        return float(np.clip(reward, -5.0, 5.0))
