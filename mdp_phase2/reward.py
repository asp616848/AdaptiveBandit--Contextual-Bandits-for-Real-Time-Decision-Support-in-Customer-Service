"""
Multi-Component Reward Shaper — Phase II
=========================================
The reward function is a key design choice in RL.
Simple "+5 for resolve, -3 for escalate" is not enough —
we also want the agent to learn good intermediate behaviour.

Final reward formula (following the project's defined function):

  r_t = α·ΔSentiment  +  β·Resolved  −  γ·Escalated  −  δ·steps  +  bonuses  −  penalties

Shaping bonuses / penalties:
  +  info_progress  : small reward for gathering new information
  +  sentiment_recovery : reward for improving a very negative customer
  −  repetition_penalty : penalise using the same strategy 3+ times in a row
  −  premature_close    : heavy penalty for closing without resolution

These are logged separately so training notebooks can decompose total reward
and understand what the agent is actually optimising.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


# ── Reward weight configuration ──────────────────────────────────────────────

@dataclass
class RewardConfig:
    # primary components
    alpha  : float = 2.0    # weight on ΔSentiment per step
    beta   : float = 5.0    # bonus on resolve
    gamma  : float = 3.0    # penalty on escalate
    delta  : float = 0.02   # per-step cost

    # shaping weights
    info_progress_weight        : float = 0.30
    sentiment_recovery_weight   : float = 0.50
    repetition_penalty_weight   : float = 0.15
    premature_close_penalty     : float = 2.50
    efficiency_bonus_threshold  : int   = 5     # resolve within this many turns for bonus
    efficiency_bonus_value      : float = 1.50


# ── Per-transition reward breakdown ──────────────────────────────────────────

@dataclass
class RewardBreakdown:
    """Returned alongside the scalar reward for analysis and logging."""
    total               : float = 0.0
    sentiment_component : float = 0.0
    resolve_bonus       : float = 0.0
    escalate_penalty    : float = 0.0
    step_cost           : float = 0.0
    info_progress       : float = 0.0
    sentiment_recovery  : float = 0.0
    repetition_penalty  : float = 0.0
    efficiency_bonus    : float = 0.0
    raw_env_reward      : float = 0.0   # original reward from environment.py

    def as_dict(self) -> dict:
        return {
            "total":               self.total,
            "sentiment_component": self.sentiment_component,
            "resolve_bonus":       self.resolve_bonus,
            "escalate_penalty":    self.escalate_penalty,
            "step_cost":           self.step_cost,
            "info_progress":       self.info_progress,
            "sentiment_recovery":  self.sentiment_recovery,
            "repetition_penalty":  self.repetition_penalty,
            "efficiency_bonus":    self.efficiency_bonus,
            "raw_env_reward":      self.raw_env_reward,
        }


# ── Reward shaper ─────────────────────────────────────────────────────────────

class RewardShaper:
    """
    Wraps the environment's raw reward with additional shaping signals.

    Usage
    -----
    shaper = RewardShaper()
    shaped_reward, breakdown = shaper.shape(
        raw_reward=env_reward,
        prev_state=s,
        next_state=s_prime,
        action=a,
        done=done,
        outcome=outcome,
        turn=turn,
        consecutive_count=consecutive_count,
    )
    """

    def __init__(self, config: RewardConfig | None = None):
        self.cfg = config or RewardConfig()

    def shape(
        self,
        raw_reward        : float,
        prev_state        : np.ndarray,
        next_state        : np.ndarray,
        action            : int,
        done              : bool,
        outcome           : str | None,
        turn              : int,
        consecutive_count : int,
    ) -> tuple[float, RewardBreakdown]:
        """
        Compute shaped reward.

        State layout (indices 0-3 are the core signals):
          [0] sentiment, [1] frustration, [2] info_gathered, [3] turn_norm
        """
        bd = RewardBreakdown(raw_env_reward=raw_reward)

        # ── 1. Step cost ───────────────────────────────────────────────
        bd.step_cost = -self.cfg.delta

        # ── 2. Sentiment change component ──────────────────────────────
        delta_s = next_state[0] - prev_state[0]
        bd.sentiment_component = self.cfg.alpha * delta_s

        # ── 3. Terminal bonuses / penalties ────────────────────────────
        if done and outcome == 'resolved':
            bd.resolve_bonus = self.cfg.beta
            # efficiency bonus for quick resolution
            if turn <= self.cfg.efficiency_bonus_threshold:
                bd.efficiency_bonus = self.cfg.efficiency_bonus_value
        elif done and outcome == 'escalated':
            bd.escalate_penalty = -self.cfg.gamma
        # 'abandoned' already penalised in env; no extra penalty here

        # ── 4. Info progress shaping ───────────────────────────────────
        # reward incremental information gain (mid-episode only)
        if not done:
            delta_info = next_state[2] - prev_state[2]
            if delta_info > 0:
                bd.info_progress = self.cfg.info_progress_weight * delta_info

        # ── 5. Sentiment recovery bonus ────────────────────────────────
        # extra reward when pulling a very negative customer back up
        if not done and prev_state[0] < 0.35 and delta_s > 0:
            bd.sentiment_recovery = self.cfg.sentiment_recovery_weight * delta_s

        # ── 6. Repetition penalty ──────────────────────────────────────
        # disincentivise getting stuck in one strategy
        if consecutive_count >= 3:
            bd.repetition_penalty = -self.cfg.repetition_penalty_weight * (consecutive_count - 2)

        # ── Total ──────────────────────────────────────────────────────
        bd.total = (
            bd.step_cost
            + bd.sentiment_component
            + bd.resolve_bonus
            + bd.escalate_penalty
            + bd.info_progress
            + bd.sentiment_recovery
            + bd.repetition_penalty
            + bd.efficiency_bonus
        )
        return bd.total, bd


# ── Convenience: episode statistics accumulator ───────────────────────────────

class EpisodeStats:
    """Accumulates stats across one conversation episode."""

    def __init__(self):
        self.rewards: list[float] = []
        self.breakdowns: list[RewardBreakdown] = []
        self.actions: list[int] = []
        self.outcome: str | None = None
        self.turns: int = 0

    def record(self, reward: float, breakdown: RewardBreakdown, action: int):
        self.rewards.append(reward)
        self.breakdowns.append(breakdown)
        self.actions.append(action)
        self.turns += 1

    def total_reward(self) -> float:
        return sum(self.rewards)

    def strategy_distribution(self) -> dict:
        from collections import Counter
        from strategy_prompts import STRATEGY_NAMES
        counts = Counter(self.actions)
        return {STRATEGY_NAMES[i]: counts.get(i, 0) for i in range(len(STRATEGY_NAMES))}

    def summary(self) -> dict:
        return {
            "total_reward":          self.total_reward(),
            "turns":                 self.turns,
            "outcome":               self.outcome,
            "mean_reward_per_turn":  np.mean(self.rewards) if self.rewards else 0.0,
            "strategy_distribution": self.strategy_distribution(),
        }


if __name__ == "__main__":
    shaper = RewardShaper()
    prev = np.array([0.3, 0.7, 0.2, 0.1] + [0]*11)
    nxt  = np.array([0.5, 0.5, 0.4, 0.2] + [0]*11)
    r, bd = shaper.shape(
        raw_reward=0.2, prev_state=prev, next_state=nxt,
        action=2, done=False, outcome=None,
        turn=3, consecutive_count=1,
    )
    print(f"Shaped reward: {r:.4f}")
    for k, v in bd.as_dict().items():
        print(f"  {k:<25s}: {v:+.4f}")
