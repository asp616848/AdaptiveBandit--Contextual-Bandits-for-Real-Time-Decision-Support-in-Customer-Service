from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from simulation_core.config import ACTION_SPACE_7, DEFAULT_TIER_DISTRIBUTION, TIER_VALUES

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - runtime optional
    torch = None
    nn = None

try:
    from transformers import pipeline as hf_pipeline
except Exception:  # pragma: no cover - runtime optional
    hf_pipeline = None


class NeuralTransitionModel(nn.Module):
    def __init__(self, z_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + len(ACTION_SPACE_7) + z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class BeliefState:
    frustration: float = 0.5
    sentiment: float = 0.0
    escalation_risk: float = 0.1


class CustomerSupportPOMDP:
    """
    POMDP environment where LLM/RAG is only used in reset-like initialization,
    while per-step transitions are driven by a lightweight neural model.
    """

    def __init__(self, transition_model: Optional[NeuralTransitionModel] = None, max_turns: int = 12):
        self.action_space = ACTION_SPACE_7
        self.max_turns = max_turns
        self.transition_model = transition_model

        self.sentiment_estimator = None
        if hf_pipeline is not None:
            try:
                self.sentiment_estimator = hf_pipeline("sentiment-analysis")
            except Exception:
                self.sentiment_estimator = None

        self.history: List[Dict[str, str]] = []
        self.turn_index = 0
        self.tier = "Pro"

        self.true_frustration = 0.5
        self.true_sentiment = 0.0
        self.true_intent = "resolution_seeking"
        self.escalated = False
        self.resolved = False

        self.belief = BeliefState()
        self.retrieved_contexts: List[str] = []

    def _action_one_hot(self, action_idx: int) -> np.ndarray:
        vec = np.zeros(len(self.action_space), dtype=np.float32)
        vec[action_idx] = 1.0
        return vec

    def _estimate_text_state(self, text: str) -> Tuple[float, float]:
        if self.sentiment_estimator is None:
            lowered = (text or "").lower()
            sent = -0.2 if any(k in lowered for k in ["bad", "angry", "frustrat", "cancel"]) else 0.1
            fr = float(np.clip(0.5 - sent * 0.5, 0.0, 1.0))
            return float(sent), fr

        try:
            out = self.sentiment_estimator(text[:512])[0]
            label = out["label"].lower()
            score = float(out["score"])
            sent = score if "pos" in label else -score
            fr = float(np.clip(0.5 - sent * 0.5, 0.0, 1.0))
            return sent, fr
        except Exception:
            return 0.0, 0.5

    def _default_transition(self, action_idx: int) -> Tuple[float, float]:
        action = self.action_space[action_idx]
        delta_fr, delta_se = 0.0, 0.0
        if action == "Affective_Repair":
            delta_fr, delta_se = -0.10, 0.08
        elif action == "Provide_Solution":
            delta_fr, delta_se = -0.08, 0.10
        elif action == "Ask_for_Information":
            delta_fr, delta_se = 0.03, -0.04
        elif action == "Escalate_to_Human":
            delta_fr, delta_se = -0.15, 0.03
            self.escalated = True
        elif action == "Set_Expectation":
            delta_fr, delta_se = -0.05, 0.03
        elif action == "Proactive_Update":
            delta_fr, delta_se = -0.04, 0.04
        elif action == "Close_with_Feedback":
            delta_fr, delta_se = -0.02, 0.02
        return delta_fr, delta_se

    def reset(self, opening_user_text: Optional[str] = None, tier: Optional[str] = None, retrieved_contexts: Optional[List[str]] = None):
        self.turn_index = 0
        self.tier = tier or random.choices(TIER_VALUES, weights=DEFAULT_TIER_DISTRIBUTION, k=1)[0]
        self.history = []

        self.true_frustration = 0.6
        self.true_sentiment = -0.2
        self.true_intent = random.choice(["resolution_seeking", "venting", "info_seeking", "escalating"])
        self.escalated = False
        self.resolved = False

        opening_user_text = opening_user_text or "My issue is still not resolved. Can you help me?"
        self.history.append({"role": "customer", "text": opening_user_text})

        # RAG retrieval is provided here by caller; never used inside step().
        self.retrieved_contexts = retrieved_contexts or []

        sent, fr = self._estimate_text_state(opening_user_text)
        self.belief = BeliefState(frustration=fr, sentiment=sent, escalation_risk=0.2)
        return self._observation()

    def _observation(self) -> Dict[str, object]:
        return {
            "conversation_history": self.history[-5:],
            "belief_frustration": float(self.belief.frustration),
            "belief_sentiment": float(self.belief.sentiment),
            "belief_escalation_risk": float(self.belief.escalation_risk),
            "turn_index": int(self.turn_index),
            "tier": self.tier,
        }

    def _simulate_customer_text(self, action: str) -> str:
        if action == "Provide_Solution":
            return "I tried your steps and this is better now."
        if action == "Affective_Repair":
            return "Thanks for understanding, that helps."
        if action == "Escalate_to_Human":
            return "Please transfer me to a specialist now."
        if action == "Ask_for_Information":
            return "I already shared this before and this is frustrating."
        if action == "Set_Expectation":
            return "Okay, I can wait if there is a clear timeline."
        if action == "Proactive_Update":
            return "Thanks for the update, I appreciate it."
        return "All right, anything else I should do?"

    def step(self, action_idx: int):
        self.turn_index += 1
        action = self.action_space[action_idx]
        self.history.append({"role": "agent", "text": action})

        if self.transition_model is not None and torch is not None:
            vec = np.concatenate(
                [
                    np.array([self.true_frustration, self.true_sentiment], dtype=np.float32),
                    self._action_one_hot(action_idx),
                    np.zeros(8, dtype=np.float32),
                ]
            )
            with torch.no_grad():
                delta = self.transition_model(torch.tensor(vec).unsqueeze(0)).squeeze(0).numpy()
            delta_fr, delta_se = float(delta[0]), float(delta[1])
        else:
            delta_fr, delta_se = self._default_transition(action_idx)

        self.true_frustration = float(np.clip(self.true_frustration + delta_fr, 0.0, 1.0))
        self.true_sentiment = float(np.clip(self.true_sentiment + delta_se, -1.0, 1.0))

        customer_text = self._simulate_customer_text(action)
        self.history.append({"role": "customer", "text": customer_text})
        observed_sent, observed_fr = self._estimate_text_state(customer_text)

        alpha = 0.3
        self.belief.frustration = alpha * observed_fr + (1 - alpha) * self.belief.frustration
        self.belief.sentiment = alpha * observed_sent + (1 - alpha) * self.belief.sentiment
        self.belief.escalation_risk = float(np.clip(0.7 * self.belief.frustration + 0.3 * (1.0 - max(self.belief.sentiment, 0.0)), 0.0, 1.0))

        self.resolved = self.true_sentiment > 0.5 and self.true_frustration < 0.3
        done = self.resolved or self.escalated or self.turn_index >= self.max_turns

        reward = 1.5 * (self.belief.sentiment) - 1.2 * (self.belief.frustration)
        if self.resolved:
            reward += 4.0
        if self.escalated:
            reward -= 2.0

        info = {
            "resolved": self.resolved,
            "escalated": self.escalated,
            "retrieved_contexts": self.retrieved_contexts[:2],
        }
        return self._observation(), float(reward), bool(done), info


def train_transition_model(turn_pairs_df: pd.DataFrame, epochs: int = 20) -> Optional[NeuralTransitionModel]:
    if torch is None or nn is None:
        return None
    if turn_pairs_df.empty:
        return None

    model = NeuralTransitionModel(z_dim=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    x_rows, y_rows = [], []
    for _, r in turn_pairs_df.iterrows():
        x = np.concatenate(
            [
                np.array([r["frustration_t"], r["sentiment_t"]], dtype=np.float32),
                r["action_one_hot"],
                np.zeros(8, dtype=np.float32),
            ]
        )
        y = np.array([r["delta_frustration"], r["delta_sentiment"]], dtype=np.float32)
        x_rows.append(x)
        y_rows.append(y)

    X = torch.tensor(np.stack(x_rows), dtype=torch.float32)
    Y = torch.tensor(np.stack(y_rows), dtype=torch.float32)

    for _ in range(epochs):
        opt.zero_grad()
        pred = model(X)
        loss = mse(pred, Y)
        loss.backward()
        opt.step()

    return model
