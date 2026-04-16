from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class TextOnlyObservationWrapper(gym.Wrapper):
    """Expose only text-derived observations to the RL policy.

    The wrapped policy never sees hidden simulator state variables (for example,
    frustration, progress, probabilities, persona, or tier). It only receives a
    fixed-size vector built from textual conversation context.
    """

    ACTION_NAME_TO_TEXT = {
        "AskInfo": "Could you share the missing details so I can help?",
        "ProvideSolution": "Please try this fix and tell me if it works.",
        "AffectiveRepair": "I understand this is frustrating, and I will help.",
        "Escalate": "I am escalating this to a specialist now.",
        "Close": "We can close this case if everything is resolved.",
    }

    def __init__(self, env: gym.Env, n_features: int = 512):
        super().__init__(env)
        self.n_features = int(n_features)
        self.vectorizer = HashingVectorizer(
            n_features=self.n_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
        )
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_features,),
            dtype=np.float32,
        )
        self._last_action_name = "None"
        self._turn_index = 0

    @property
    def state(self) -> dict[str, Any]:
        return getattr(self.env, "state", {})

    @property
    def ACTION_NAMES(self) -> dict[int, str]:
        return getattr(self.env, "ACTION_NAMES", {})

    def set_subflow_filter(self, subflow_filter: list[str] | None) -> None:
        if hasattr(self.env, "set_subflow_filter"):
            self.env.set_subflow_filter(subflow_filter)

    def _latest_customer_text(self, info: dict[str, Any]) -> str:
        transition = info.get("transition_outcome", {})
        if isinstance(transition, dict):
            utt = str(transition.get("customer_utterance", "")).strip()
            if utt:
                return utt

        history = info.get("conversation_history", [])
        if isinstance(history, list):
            for msg in reversed(history):
                if not isinstance(msg, dict):
                    continue
                if str(msg.get("role", "")).strip() == "assistant":
                    text = str(msg.get("content", "")).strip()
                    if text:
                        return text

        return ""

    def _transition_summary(self, info: dict[str, Any]) -> str:
        transition = info.get("transition_outcome", {})
        if not isinstance(transition, dict):
            return ""

        outcome = str(transition.get("outcome", "")).strip()
        terminal = str(transition.get("terminal_type", "")).strip()
        slots_revealed = transition.get("slots_revealed", [])

        pieces: list[str] = []
        if outcome:
            pieces.append(f"outcome={outcome}")
        if terminal:
            pieces.append(f"terminal={terminal}")

        if isinstance(slots_revealed, list) and slots_revealed:
            slot_keys: list[str] = []
            for item in slots_revealed:
                if isinstance(item, (list, tuple)) and item:
                    slot_keys.append(str(item[0]))
            if slot_keys:
                pieces.append("revealed=" + ",".join(slot_keys[:4]))

        return " ; ".join(pieces)

    def _build_text_observation(self, info: dict[str, Any]) -> np.ndarray:
        customer_text = self._latest_customer_text(info)
        transition_text = self._transition_summary(info)

        if not customer_text:
            if self._turn_index == 0:
                customer_text = "Conversation just started. Customer issue is not yet described."
            else:
                customer_text = "Customer gave no additional text."

        text = (
            f"turn={self._turn_index} ; "
            f"last_agent_action={self._last_action_name} ; "
            f"customer_text={customer_text} ; "
            f"transition={transition_text}"
        )

        vec = self.vectorizer.transform([text])
        arr = vec.toarray().astype(np.float32)[0]
        return arr

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        _obs, info = self.env.reset(seed=seed, options=options)
        self._last_action_name = "None"
        self._turn_index = 0
        text_obs = self._build_text_observation(info)
        return text_obs, info

    def step(self, action: int):
        action_name = self.ACTION_NAMES.get(int(action), f"Action{int(action)}")
        agent_text = self.ACTION_NAME_TO_TEXT.get(action_name, f"Agent action: {action_name}")

        obs, reward, done, truncated, info = self.env.step(int(action), agent_text=agent_text)
        _ = obs

        self._last_action_name = action_name
        self._turn_index += 1
        text_obs = self._build_text_observation(info)
        return text_obs, reward, done, truncated, info
