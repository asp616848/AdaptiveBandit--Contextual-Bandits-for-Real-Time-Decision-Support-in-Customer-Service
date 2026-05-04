from __future__ import annotations

"""
NLP Observation Wrapper — Stage 3.

Replaces the raw simulator state (or hashed text) with a 9D observation built
from LLM-classified intent features plus three legitimately observable state
variables.  The PPO policy then learns to act on semantic conversation features
rather than hidden simulator internals.

Observation vector (9D):
  [0] intent_norm        — classified intent / (n_intents-1), in [0,1]
  [1] confidence         — LLM confidence in the intent classification
  [2] sentiment_norm     — frustrated=0, neutral=0.5, satisfied=1.0
  [3] suggested_action_norm — LLM hint: which action fits (0=AskInfo … 1=Close)
  [4] escalation_flag    — 1 if LLM thinks escalation is needed, else 0
  [5] info_completeness  — LLM estimate of how complete the customer's info is
    [6] turn_count_norm    — turn_count / T_max (tracked by wrapper)
    [7] history_depth_norm — normalised length of conversation history
    [8] customer_len_norm  — latest customer message length proxy

The wrapper also calls AgentResponseGenerator (if available) to produce natural-
language agent text for each action before calling env.step().  This feeds the
conversation with real sentences, enabling more realistic NLG training.
"""

from typing import Any

import gymnasium as gym
import numpy as np

from Simulation_4.env.intent_classifier import IntentClassifier

NLP_OBS_DIM = 9


class NLPObservationWrapper(gym.Wrapper):
    """Observation wrapper that uses LLM intent classification as the RL state."""

    # Canonical agent response templates used as fallback when
    # AgentResponseGenerator is not available or Ollama is down.
    ACTION_NAME_TO_TEXT = {
        "AskInfo": "Could you share the missing details so I can help you resolve this?",
        "ProvideSolution": "Please try the following steps and let me know if this resolves the issue.",
        "AffectiveRepair": "I completely understand how frustrating this must be. I am here to help.",
        "Escalate": "I am going to connect you with a specialist who can best assist you with this.",
        "Close": "I am glad we could resolve this for you today. Is there anything else I can help with?",
    }

    def __init__(
        self,
        env: gym.Env,
        intent_model: str | None = None,
        agent_response_generator=None,  # Optional AgentResponseGenerator instance
    ):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(NLP_OBS_DIM,),
            dtype=np.float32,
        )
        self.intent_classifier = IntentClassifier(model=intent_model)
        self.agent_response_generator = agent_response_generator
        self._last_classification: dict[str, Any] = {}
        self._conversation_history: list[dict[str, str]] = []

    @property
    def state(self) -> dict[str, Any]:
        return getattr(self.env, "state", {})

    @property
    def ACTION_NAMES(self) -> dict[int, str]:
        return getattr(self.env, "ACTION_NAMES", {})

    def set_subflow_filter(self, subflow_filter: list[str] | None) -> None:
        if hasattr(self.env, "set_subflow_filter"):
            self.env.set_subflow_filter(subflow_filter)

    def _get_policy_context(self) -> str:
        rag = getattr(self.env, "rag_context", None)
        if rag and isinstance(rag, dict):
            return str(rag.get("policy_context", ""))
        return ""

    def _latest_customer_text(self, info: dict[str, Any]) -> str:
        history = info.get("conversation_history", self._conversation_history)
        if isinstance(history, list):
            for msg in reversed(history):
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "")).strip()
                if role == "assistant":
                    return str(msg.get("content", "")).strip()
        return ""

    def _build_obs(self, info: dict[str, Any]) -> np.ndarray:
        history = info.get("conversation_history", self._conversation_history)

        classification = self.intent_classifier.classify(
            conversation_history=history,
            policy_context=self._get_policy_context(),
            subflow="",
        )
        self._last_classification = classification
        intent_features = self.intent_classifier.to_feature_vector(classification)

        turn_norm = float(min(len(history) // 2, 20)) / 20.0 if isinstance(history, list) else 0.0
        history_depth_norm = float(min(len(history), 40)) / 40.0 if isinstance(history, list) else 0.0
        customer_len_norm = float(min(len(self._latest_customer_text(info)), 300)) / 300.0

        obs = np.array(intent_features + [turn_norm, history_depth_norm, customer_len_norm], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)

    def _generate_agent_text(self, action_name: str, info: dict[str, Any]) -> str:
        """Generate a natural language agent response for the chosen action."""
        if self.agent_response_generator is not None:
            try:
                history = info.get("conversation_history", self._conversation_history)
                return self.agent_response_generator.generate(
                    action_name=action_name,
                    conversation_history=history,
                    policy_context=self._get_policy_context(),
                    classification=self._last_classification,
                )
            except Exception:
                pass
        return self.ACTION_NAME_TO_TEXT.get(action_name, f"Agent action: {action_name}")

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        _obs, info = self.env.reset(seed=seed, options=options)
        self._conversation_history = info.get("conversation_history", [])
        self._last_classification = {}
        obs = self._build_obs(info)
        return obs, info

    def step(self, action: int):
        action_name = self.ACTION_NAMES.get(int(action), f"Action{int(action)}")

        # Build agent text BEFORE calling env.step() so it is passed to NLG.
        # We use the last known info to generate a contextually appropriate response.
        agent_text = self._generate_agent_text(action_name, {
            "conversation_history": self._conversation_history,
        })

        obs, reward, done, truncated, info = self.env.step(int(action), agent_text=agent_text)
        _ = obs  # base env obs is not used

        self._conversation_history = info.get("conversation_history", self._conversation_history)

        nlp_obs = self._build_obs(info)
        return nlp_obs, reward, done, truncated, info
