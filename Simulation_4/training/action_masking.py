from __future__ import annotations

import numpy as np
from gymnasium import Wrapper


class ActionMaskedEnv(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def action_masks(self) -> np.ndarray:
        mask = np.ones(5, dtype=bool)
        # No masking at all: let reward and terminal penalties shape behavior.
        return mask

    def valid_action_mask(self) -> np.ndarray:
        return self.action_masks()

    def step(self, action):
        mask = self.action_masks()
        if int(action) < 0 or int(action) >= len(mask) or not bool(mask[int(action)]):
            action = 0
        return self.env.step(action)
