from __future__ import annotations

import numpy as np
from gymnasium import Wrapper


class ActionMaskedEnv(Wrapper):
    """Thin wrapper kept for API compatibility. All actions always valid.

    Previous implementation blocked Escalate (action 3) in turns 0-2, then
    replaced masked actions with action 0 before calling env.step(). This
    corrupted PPO training: the buffer stored (obs, action=3, reward_of_action_0),
    giving escalation wrong Q-value estimates and driving its policy probability
    to zero regardless of reward. Removed. The reward function already penalises
    inappropriate escalation (fresh-state cost up to -1.5); no masking needed.
    """

    def __init__(self, env):
        super().__init__(env)

    @property
    def ACTION_NAMES(self):
        return self.env.ACTION_NAMES

    @property
    def state(self):
        return self.env.state

    def action_masks(self) -> np.ndarray:
        return np.ones(5, dtype=bool)

    def valid_action_mask(self) -> np.ndarray:
        return self.action_masks()

    def step(self, action):
        return self.env.step(action)
