from __future__ import annotations

import numpy as np
from gymnasium import Wrapper


class ActionMaskedEnv(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    @property
    def ACTION_NAMES(self):
        return self.env.ACTION_NAMES

    @property
    def state(self):
        return self.env.state

    def action_masks(self) -> np.ndarray:
        mask = np.ones(5, dtype=bool)
        # Block Escalate (action 3) in the first 3 turns to prevent the
        # "escalate immediately" degenerate policy that appears when the
        # agent cannot yet distinguish episode context.
        state = getattr(self.env, "state", {})
        turn_count = int(state.get("turn_count", 0))
        if turn_count < 3:
            mask[3] = False
        return mask

    def valid_action_mask(self) -> np.ndarray:
        return self.action_masks()

    def step(self, action):
        mask = self.action_masks()
        if int(action) < 0 or int(action) >= len(mask) or not bool(mask[int(action)]):
            action = 0
        return self.env.step(action)
