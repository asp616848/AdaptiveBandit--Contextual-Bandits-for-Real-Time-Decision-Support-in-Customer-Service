from __future__ import annotations

import math
from typing import Any

import numpy as np


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-x)))


class StateEngine:
    """Transition dynamics engine for SupportEnv."""

    def __init__(self, transition_params: dict[str, Any], psuccess_params: dict[str, Any]):
        self.p_info_gain = float(transition_params["askinfo"]["p_info_gain"])
        self.mu_ask_conditional = float(transition_params["askinfo"]["mu_ask_conditional"])
        self.mu_ask_conditional_by_persona = transition_params["askinfo"].get("mu_ask_conditional_by_persona", {})
        self.ask_noise_std = float(transition_params["askinfo"].get("noise_std", 0.02))

        self.p_repair_base = float(transition_params["affective_repair"]["p_repair_base_design"])
        self.close_threshold = float(transition_params["close"]["close_score_threshold"])
        self.auto_resolve_threshold = float(transition_params["progress"]["auto_resolve_threshold"])

        self.dropout_coeffs = self._parse_dropout_coefficients(transition_params["dropout"])

        self.theta_0 = float(psuccess_params["coefficients"]["theta_0"])
        self.theta_i = float(psuccess_params["coefficients"]["theta_i"])
        self.subflow_offsets = dict(psuccess_params.get("subflow_offsets", {}))
        self.action_offsets = dict(psuccess_params.get("action_offsets", {}))
        self.subflow_mean_values: dict[str, float] = {}

    def _parse_dropout_coefficients(self, dropout_block: dict[str, Any]) -> dict[str, float]:
        if "coefficients" in dropout_block and isinstance(dropout_block["coefficients"], dict):
            coeffs = dropout_block["coefficients"]
            return {
                "c0": float(coeffs.get("c0", -5.0)),
                "cf": float(coeffs.get("cf", 3.5)),
                "cs": float(coeffs.get("cs", 0.3)),
                "ct": float(coeffs.get("ct", 0.1)),
                "ctau": float(coeffs.get("ctau", -2.0)),
            }

        return {"c0": -5.0, "cf": 3.5, "cs": 0.3, "ct": 0.1, "ctau": -2.0}

    def compute_p_success(self, state: dict[str, Any], action_name: str = "ProvideSolution") -> float:
        info = float(state.get("information", 0.0))
        subflow = str(state.get("subflow", ""))
        alpha_subflow = float(self.subflow_offsets.get(subflow, 0.0))
        alpha_action = float(self.action_offsets.get(action_name, 0.0))
        z = self.theta_0 + self.theta_i * info + alpha_subflow + alpha_action
        return sigmoid(z)

    def compute_p_dropout(self, state: dict[str, Any]) -> float:
        f = float(state.get("frustration", 0.0))
        streak = int(state.get("failed_streak", 0))
        turns = int(state.get("turn_count", 0))
        tau = float(state.get("tau", 0.5))
        c = self.dropout_coeffs
        z = c["c0"] + c["cf"] * f + c["cs"] * streak + c["ct"] * turns + c["ctau"] * tau
        return sigmoid(z)

    def transition_ask_info(self, state: dict[str, Any], rng: np.random.Generator) -> tuple[dict[str, Any], dict[str, Any]]:
        i_t = float(state["information"])
        d = float(state["difficulty"])
        rho = float(state["rho"])
        persona = str(state.get("persona_label", ""))
        subflow = str(state.get("subflow", ""))

        p_gain = self.p_info_gain * (0.6 + 0.8 * rho)
        p_gain = float(np.clip(p_gain, 0.0, 1.0))
        gain_occurs = bool(rng.random() < p_gain)

        if gain_occurs:
            persona_mu = float(self.mu_ask_conditional_by_persona.get(persona, self.mu_ask_conditional))
            noise = float(rng.normal(0.0, self.ask_noise_std))
            delta_i = float(np.clip(persona_mu * (1.0 - 0.4 * d) + noise, 0.0, 1.0 - i_t))
        else:
            delta_i = 0.0

        state["information"] = float(np.clip(i_t + delta_i, 0.0, 1.0))
        state["progress"] = float(np.clip(float(state["progress"]) + 0.05 * delta_i, 0.0, 1.0))

        streak = int(state["failed_streak"])
        delta_f = 0.03 * (1.0 - rho) + 0.01 * d - 0.06 * abs(delta_i) + 0.03 * max(streak - 1, 0)
        state["frustration"] = float(np.clip(float(state["frustration"]) + delta_f, 0.0, 1.0))

        if gain_occurs and delta_i > 0.0:
            mean_vals = float(self.subflow_mean_values.get(subflow, 3.0))
            n_slots_to_reveal = max(1, int(round(delta_i * mean_vals)))
            slots_to_reveal = int(n_slots_to_reveal)
        else:
            slots_to_reveal = 0

        outcome = {
            "action": "AskInfo",
            "gain_occurs": gain_occurs,
            "gain_occurred": gain_occurs,
            "delta_i": delta_i,
            "delta_cumulative_values": delta_i,
            "slots_to_reveal": slots_to_reveal,
            "outcome": "gain" if gain_occurs else "no_gain",
            "terminal_type": None,
        }
        return state, outcome

    def transition_provide_solution(self, state: dict[str, Any], rng: np.random.Generator) -> tuple[dict[str, Any], dict[str, Any]]:
        p_success = self.compute_p_success(state, action_name="ProvideSolution")
        success = bool(rng.random() < p_success)

        if success:
            info = float(state["information"])
            progress_gain = float(np.clip(0.25 + 0.20 * info, 0.0, 1.0 - float(state["progress"])))
            state["progress"] = float(np.clip(float(state["progress"]) + progress_gain, 0.0, 1.0))

            frustration_drop = float(np.clip(0.20 + 0.10 * info, 0.0, float(state["frustration"])))
            state["frustration"] = float(np.clip(float(state["frustration"]) - frustration_drop, 0.0, 1.0))
            state["failed_streak"] = 0

            if float(state["progress"]) >= self.auto_resolve_threshold:
                state["resolved"] = 1
                state["done"] = True

            outcome = {
                "action": "ProvideSolution",
                "outcome": "success",
                "p_success_used": float(p_success),
                "terminal_type": "success" if state["done"] else None,
            }
            return state, outcome

        state["failed_streak"] = int(state["failed_streak"]) + 1
        sigma = float(state["sigma"])
        tau = float(state["tau"])
        streak = int(state["failed_streak"])
        delta_f = 0.05 + 0.15 * sigma + 0.10 * (streak / (streak + 3.0)) * (1.0 - tau)
        state["frustration"] = float(np.clip(float(state["frustration"]) + delta_f, 0.0, 1.0))

        outcome = {
            "action": "ProvideSolution",
            "outcome": "failure",
            "p_success_used": float(p_success),
            "terminal_type": None,
        }
        return state, outcome

    def transition_affective_repair(self, state: dict[str, Any], rng: np.random.Generator) -> tuple[dict[str, Any], dict[str, Any]]:
        rho = float(state["rho"])
        sigma = float(state["sigma"])
        p_repair = float(np.clip(self.p_repair_base + 0.4 * rho, 0.0, 1.0))
        repair_effective = bool(rng.random() < p_repair)

        if repair_effective:
            drop = 0.15 * (0.6 + 0.6 * sigma)
            state["frustration"] = float(np.clip(float(state["frustration"]) - drop, 0.0, 1.0))
        else:
            state["frustration"] = float(np.clip(float(state["frustration"]) + 0.02, 0.0, 1.0))

        outcome = {
            "action": "AffectiveRepair",
            "repair_effective": repair_effective,
            "p_repair_used": float(p_repair),
            "outcome": "effective" if repair_effective else "ineffective",
            "terminal_type": None,
        }
        return state, outcome

    def transition_escalate(self, state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        state["escalated"] = 1
        state["resolved"] = 0
        state["done"] = True
        outcome = {
            "action": "Escalate",
            "outcome": "escalated",
            "terminal_type": "escalation",
            "resolved": False,
        }
        return state, outcome

    def transition_close(self, state: dict[str, Any], rng: np.random.Generator) -> tuple[dict[str, Any], dict[str, Any]]:
        _ = rng
        p_success = self.compute_p_success(state, action_name="ProvideSolution")
        close_score = (
            0.45 * p_success
            + 0.35 * float(state["progress"])
            + 0.20 * float(state["information"])
            - 0.25 * float(state["frustration"])
        )
        resolved = int(close_score >= self.close_threshold)
        state["resolved"] = resolved
        state["done"] = True
        outcome = {
            "action": "Close",
            "outcome": "success" if resolved else "failure",
            "resolved": bool(resolved),
            "p_success_used": float(p_success),
            "close_score": float(close_score),
            "terminal_type": "success" if resolved else "timeout",
        }
        return state, outcome

    def transition_autonomous_dropout(self, state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        state["dropped_off"] = 1
        state["resolved"] = 0
        state["done"] = True
        outcome = {"outcome": "dropout", "terminal_type": "dropout"}
        return state, outcome

    def advance_turn_and_apply_timeout(self, state: dict[str, Any], t_max: int) -> tuple[dict[str, Any], dict[str, Any]]:
        state["turn_count"] = int(state["turn_count"]) + 1
        outcome: dict[str, Any] = {}
        if int(state["turn_count"]) >= int(t_max) and not bool(state["done"]):
            state["done"] = True
            outcome["terminal_type"] = "timeout"
            outcome["outcome"] = "timeout"
        return state, outcome
