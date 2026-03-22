from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from simulation_core.config import ACTION_SPACE_7

ACTION_TO_ID = {a: i for i, a in enumerate(ACTION_SPACE_7)}
TIER_TO_ID = {"Enterprise": 0, "Business+": 1, "Pro": 2, "Free": 3}


def phi(state: Dict[str, float], action_label: str) -> np.ndarray:
    action_one_hot = np.zeros(len(ACTION_SPACE_7), dtype=float)
    action_one_hot[ACTION_TO_ID.get(action_label, 1)] = 1.0

    tier_one_hot = np.zeros(4, dtype=float)
    tier_one_hot[TIER_TO_ID.get(state.get("tier", "Pro"), 2)] = 1.0

    feat = np.concatenate(
        [
            np.array(
                [
                    state.get("frustration", 0.5),
                    state.get("sentiment", 0.0),
                    state.get("turn_index_norm", 0.2),
                    state.get("escalation_risk", 0.1),
                ],
                dtype=float,
            ),
            action_one_hot,
            tier_one_hot,
        ]
    )
    return feat


@dataclass
class LinearRewardModel:
    weights: np.ndarray

    def score(self, state: Dict[str, float], action_label: str) -> float:
        return float(np.dot(self.weights, phi(state, action_label)))


class MaxEntIRLTrainer:
    def __init__(self, feature_dim: int, lr: float = 1e-3, l2_lambda: float = 0.01):
        self.weights = np.zeros(feature_dim, dtype=float)
        self.lr = lr
        self.l2_lambda = l2_lambda

    def _expected_features(self, states: List[Dict[str, float]]) -> np.ndarray:
        total = np.zeros_like(self.weights)
        for s in states:
            logits = np.array([np.dot(self.weights, phi(s, a)) for a in ACTION_SPACE_7])
            exps = np.exp(logits - logits.max())
            probs = exps / exps.sum()
            for p, a in zip(probs, ACTION_SPACE_7):
                total += p * phi(s, a)
        return total / max(len(states), 1)

    def fit(self, expert_states: List[Dict[str, float]], expert_actions: List[str], epochs: int = 100) -> LinearRewardModel:
        expert_feat = np.mean([phi(s, a) for s, a in zip(expert_states, expert_actions)], axis=0)
        for _ in range(epochs):
            model_feat = self._expected_features(expert_states)
            grad = expert_feat - model_feat - self.l2_lambda * self.weights
            self.weights += self.lr * grad
        return LinearRewardModel(weights=self.weights.copy())


class PreferenceRewardTrainer:
    def __init__(self, feature_dim: int, lr: float = 1e-3):
        self.w = np.zeros(feature_dim, dtype=float)
        self.lr = lr

    def fit(self, pair_rows: List[Tuple[np.ndarray, np.ndarray, int]], epochs: int = 50) -> LinearRewardModel:
        for _ in range(epochs):
            for a_feat, b_feat, pref in pair_rows:
                ra = float(np.dot(self.w, a_feat))
                rb = float(np.dot(self.w, b_feat))
                pa = 1.0 / (1.0 + np.exp(-(ra - rb)))
                y = 1.0 if pref == 1 else 0.0
                grad = (y - pa) * (a_feat - b_feat)
                self.w += self.lr * grad
        return LinearRewardModel(weights=self.w.copy())


def _row_to_state(r: pd.Series) -> Dict[str, float]:
    return {
        "frustration": float(r.get("frustration_score", 0.5) if pd.notna(r.get("frustration_score", np.nan)) else 0.5),
        "sentiment": float(r.get("sentiment_score", 0.0) if pd.notna(r.get("sentiment_score", np.nan)) else 0.0),
        "turn_index_norm": float(r.get("turn_index", 1)) / max(float(r.get("conv_length", 10)), 1.0),
        "tier": str(r.get("tier", "Pro")),
        "escalation_risk": float(r.get("frustration_score", 0.5) if pd.notna(r.get("frustration_score", np.nan)) else 0.5),
    }


def train_reward_model(labeled_df: pd.DataFrame, out_dir: Path, min_expert_turns: int = 100) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_dim = len(phi({"tier": "Pro"}, "Provide_Solution"))

    high_conf = labeled_df[(labeled_df["speaker_role"] == "agent") & (labeled_df["annotator_confidence"].fillna(0.0) >= 0.65)].copy()
    states = [_row_to_state(r) for _, r in high_conf.iterrows()]
    actions = [str(r.get("action_label", "Provide_Solution")) for _, r in high_conf.iterrows()]

    mode = "maxent_irl" if len(high_conf) >= min_expert_turns else "preference"

    if mode == "maxent_irl":
        trainer = MaxEntIRLTrainer(feature_dim=feature_dim)
        model = trainer.fit(states, actions, epochs=100)
    else:
        pairs = []
        rng = np.random.default_rng(42)
        for _ in range(min(200, max(10, len(high_conf) // 2))):
            a = high_conf.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
            b = high_conf.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]

            a_state = _row_to_state(a)
            b_state = _row_to_state(b)
            a_feat = phi(a_state, str(a.get("action_label", "Provide_Solution")))
            b_feat = phi(b_state, str(b.get("action_label", "Provide_Solution")))

            a_score = float(a_state["sentiment"] - a_state["frustration"])
            b_score = float(b_state["sentiment"] - b_state["frustration"])
            pref = 1 if a_score >= b_score else 0
            pairs.append((a_feat, b_feat, pref))

        model = PreferenceRewardTrainer(feature_dim=feature_dim).fit(pairs)

    # Validation 1: expert > random
    expert_scores = np.array([model.score(s, a) for s, a in zip(states, actions)], dtype=float)
    random_actions = np.random.choice(ACTION_SPACE_7, size=len(states), replace=True)
    random_scores = np.array([model.score(s, a) for s, a in zip(states, random_actions)], dtype=float)

    # Wilcoxon requires paired arrays with same shape.
    if len(expert_scores) > 0:
        pvalue = float(wilcoxon(expert_scores, random_scores, zero_method="zsplit").pvalue)
    else:
        pvalue = 1.0

    # Validation 2: affective repair in high frustration
    high_fr = [s for s in states if s["frustration"] > 0.7]
    if high_fr:
        affective = float(np.mean([model.score(s, "Affective_Repair") for s in high_fr]))
        ask_info = float(np.mean([model.score(s, "Ask_for_Information") for s in high_fr]))
    else:
        affective, ask_info = 0.0, 0.0

    result = {
        "mode": mode,
        "n_expert_turns": int(len(high_conf)),
        "mean_R_expert": float(expert_scores.mean()) if len(expert_scores) else 0.0,
        "mean_R_random": float(random_scores.mean()) if len(random_scores) else 0.0,
        "wilcoxon_pvalue": pvalue,
        "affective_repair_high_frustration": affective,
        "ask_for_information_high_frustration": ask_info,
        "expert_beats_random": bool((expert_scores.mean() if len(expert_scores) else 0.0) > (random_scores.mean() if len(random_scores) else 0.0)),
    }

    np.save(out_dir / "reward_weights.npy", model.weights)
    (out_dir / "reward_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_reward_model(weights_path: Path) -> LinearRewardModel:
    weights = np.load(weights_path)
    return LinearRewardModel(weights=weights)
