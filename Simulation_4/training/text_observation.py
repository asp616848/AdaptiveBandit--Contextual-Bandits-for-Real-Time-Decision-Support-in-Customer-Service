from __future__ import annotations

"""
Text-based observation wrapper — Stage 2 version.

Replaces the original HashingVectorizer (512D, random sparse hashes) with
sentence-transformers/all-MiniLM-L6-v2 (384D, dense semantic embeddings).
The same model is already used by LumoRAG, so no new dependency is added.

Two extra floats are appended to the embedding:
  [384] tier_norm     — customer tier (0=Free … 1=Enterprise), legitimately
                        observable since the agent knows who they are talking to
  [385] turn_count_norm — normalised turn counter, equally observable

Total observation dimension: 386.

A simple LRU text cache avoids redundant model calls for repeated inputs.
"""

from functools import lru_cache
from typing import Any

import gymnasium as gym
import numpy as np

_EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension
OBS_DIM = _EMBED_DIM + 2  # + tier_norm + turn_count_norm


def _load_model():
    """Lazy-load SentenceTransformer so import cost is paid once."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


# Module-level singleton — shared across all wrapper instances in a process.
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = _load_model()
    return _model


@lru_cache(maxsize=2048)
def _embed_text(text: str) -> tuple:
    """Embed a string and return a tuple (hashable for lru_cache)."""
    model = _get_model()
    if model is None:
        return tuple(np.zeros(_EMBED_DIM, dtype=np.float32).tolist())
    vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return tuple(float(x) for x in vec)


class TextOnlyObservationWrapper(gym.Wrapper):
    """Expose only text-derived observations to the RL policy.

    The wrapped policy never sees hidden simulator state variables (frustration,
    progress, persona). It receives a 386-dimensional vector:
      - [0:384] sentence-transformer embedding of conversation context
      - [384]   tier_norm (legitimately observable)
      - [385]   turn_count_norm (legitimately observable)
    """

    ACTION_NAME_TO_TEXT = {
        "AskInfo": "Could you share the missing details so I can help?",
        "ProvideSolution": "Please try this fix and tell me if it works.",
        "AffectiveRepair": "I understand this is frustrating, and I will help.",
        "Escalate": "I am escalating this to a specialist now.",
        "Close": "We can close this case if everything is resolved.",
    }

    def __init__(self, env: gym.Env, n_features: int = OBS_DIM):
        super().__init__(env)
        # n_features kept as parameter for API compatibility; actual dim is OBS_DIM.
        self.n_features = OBS_DIM
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBS_DIM,),
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
            slot_keys = [str(item[0]) for item in slots_revealed if isinstance(item, (list, tuple)) and item]
            if slot_keys:
                pieces.append("revealed=" + ",".join(slot_keys[:4]))
        return " ; ".join(pieces)

    def _build_obs(self, info: dict[str, Any]) -> np.ndarray:
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

        embed = np.array(_embed_text(text), dtype=np.float32)  # shape (384,)

        # Append observable state features that the agent legitimately knows.
        state = self.state
        tier_norm = float(state.get("tier_idx", 0)) / 3.0
        turn_norm = float(state.get("turn_count", self._turn_index)) / 20.0

        obs = np.concatenate([embed, [tier_norm, turn_norm]]).astype(np.float32)
        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        _obs, info = self.env.reset(seed=seed, options=options)
        self._last_action_name = "None"
        self._turn_index = 0
        return self._build_obs(info), info

    def step(self, action: int):
        action_name = self.ACTION_NAMES.get(int(action), f"Action{int(action)}")
        agent_text = self.ACTION_NAME_TO_TEXT.get(action_name, f"Agent action: {action_name}")

        obs, reward, done, truncated, info = self.env.step(int(action), agent_text=agent_text)
        _ = obs  # base env obs is not used

        self._last_action_name = action_name
        self._turn_index += 1
        return self._build_obs(info), reward, done, truncated, info
