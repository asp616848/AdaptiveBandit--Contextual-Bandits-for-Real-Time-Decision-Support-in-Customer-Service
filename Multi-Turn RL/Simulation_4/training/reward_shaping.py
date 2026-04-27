from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym


@dataclass
class RewardShaper:
    info_gain_bonus: float = 0.05
    frustration_decrease_bonus: float = 0.03
    frustration_increase_penalty: float = -0.02
    progress_increase_bonus: float = 0.10
    enabled: bool = True
    gamma: float = 0.99
    strict_potential: bool = True

    def _potential(self, state: dict[str, Any]) -> float:
        information = float(state.get("information", 0.0))
        progress = float(state.get("progress", 0.0))
        frustration = float(state.get("frustration", 0.0))

        # Potential uses only state terms, preserving policy invariance when
        # shaping is applied as gamma * Phi(s') - Phi(s).
        frustration_weight = max(self.frustration_decrease_bonus, abs(self.frustration_increase_penalty))
        return (
            self.info_gain_bonus * information
            + self.progress_increase_bonus * progress
            - frustration_weight * frustration
        )

    def shape(
        self,
        base_reward: float,
        pre_state: dict[str, Any],
        post_state: dict[str, Any],
        action: int,
        transition_outcome: dict[str, Any],
    ) -> float:
        _ = action
        if not self.enabled:
            return float(base_reward)

        if self.strict_potential:
            shaping = self.gamma * self._potential(post_state) - self._potential(pre_state)
            return float(base_reward + shaping)

        # Optional heuristic mode (kept for ablation; disabled by default).
        shaping = 0.0

        delta_i = float(transition_outcome.get("delta_i", 0.0))
        if action == 0 and delta_i > 0.01:
            shaping += self.info_gain_bonus * delta_i

        delta_progress = float(post_state.get("progress", 0.0)) - float(pre_state.get("progress", 0.0))
        if delta_progress > 0.01:
            shaping += self.progress_increase_bonus * delta_progress

        delta_frustration = float(post_state.get("frustration", 0.0)) - float(pre_state.get("frustration", 0.0))
        if delta_frustration < -0.01:
            shaping += self.frustration_decrease_bonus * abs(delta_frustration)
        elif delta_frustration > 0.05:
            shaping += self.frustration_increase_penalty * delta_frustration

        return float(base_reward + shaping)


class RewardShapedWrapper(gym.Wrapper):
    """Thin wrapper that applies reward shaping after each environment step."""

    def __init__(self, env: gym.Env, shaper: RewardShaper):
        super().__init__(env)
        self.shaper = shaper

    def set_subflow_filter(self, subflow_filter: list[str] | None) -> None:
        if hasattr(self.env, "set_subflow_filter"):
            self.env.set_subflow_filter(subflow_filter)
            return

        # Fallback for SupportEnv directly when no setter exists.
        base = self.env
        if not hasattr(base, "_build_subflow_weights"):
            return

        stats = getattr(base, "subflow_stats", None)
        if stats is None:
            return

        base.subflow_weights = base._build_subflow_weights(stats, subflow_filter)
        base.subflow_list = list(base.subflow_weights.keys())
        base.subflow_to_idx = {s: i for i, s in enumerate(base.subflow_list)}

    def step(self, action: int):
        pre_state = dict(getattr(self.env, "state", {}))
        obs, reward, done, truncated, info = self.env.step(action)
        post_state = dict(getattr(self.env, "state", {}))
        outcome = dict(info.get("last_transition_outcome", {}) or {})

        shaped_reward = self.shaper.shape(
            base_reward=float(reward),
            pre_state=pre_state,
            post_state=post_state,
            action=int(action),
            transition_outcome=outcome,
        )
        return obs, shaped_reward, done, truncated, info
