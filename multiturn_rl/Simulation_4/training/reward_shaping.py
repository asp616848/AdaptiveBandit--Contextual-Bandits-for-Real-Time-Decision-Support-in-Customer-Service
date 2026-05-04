from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym


@dataclass
class RewardShaper:
    frustration_penalty: float = -0.15
    info_gain_bonus: float = 0.05  # Kept for signature compatibility
    frustration_decrease_bonus: float = 0.03 # Kept for signature compatibility
    frustration_increase_penalty: float = -0.02 # Kept for signature compatibility
    progress_increase_bonus: float = 0.10 # Kept for signature compatibility
    enabled: bool = True
    gamma: float = 0.99
    strict_potential: bool = True # Kept for signature compatibility, but unused

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

        shaping = 0.0
        delta_frustration = float(post_state.get("frustration", 0.0)) - float(pre_state.get("frustration", 0.0))
        
        # Apply the single, clear signal when frustration increases
        if delta_frustration > 0.0:
            shaping += self.frustration_penalty

        return float(base_reward + shaping)


class RewardShapedWrapper(gym.Wrapper):
    """Thin wrapper that applies reward shaping after each environment step."""

    def __init__(self, env: gym.Env, shaper: RewardShaper):
        super().__init__(env)
        self.shaper = shaper

    @property
    def state(self) -> dict[str, Any]:
        return getattr(self.env, "state", {})

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
