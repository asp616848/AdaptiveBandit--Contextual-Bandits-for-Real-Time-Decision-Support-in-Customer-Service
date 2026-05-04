from __future__ import annotations

from typing import Any

import numpy as np


def summarize_episode_outcomes(episode_logs: list[dict[str, Any]]) -> dict[str, int]:
    out = {"success": 0, "escalation": 0, "dropout": 0, "timeout": 0, "other": 0}
    for ep in episode_logs:
        term = str(ep.get("terminal_type", "other"))
        if term in out:
            out[term] += 1
        else:
            out["other"] += 1
    return out


def compute_bandit_metrics(episode_logs: list[dict[str, Any]], early_k: int = 3) -> dict[str, float]:
    if not episode_logs:
        return {
            "avg_per_turn_reward": 0.0,
            "early_turn_reward": 0.0,
            "sentiment_improvement": 0.0,
            "immediate_regret": 0.0,
            "resolution_rate": 0.0,
            "mean_episode_reward": 0.0,
        }

    all_turn_rewards = []
    early_turn_rewards = []
    all_sentiment_delta = []
    immediate_regret = []
    episode_rewards = []

    for ep in episode_logs:
        turns = ep.get("turns", [])
        ep_reward = 0.0
        for idx, t in enumerate(turns):
            r = float(t.get("reward", 0.0))
            all_turn_rewards.append(r)
            ep_reward += r
            if idx < early_k:
                early_turn_rewards.append(r)

            all_sentiment_delta.append(float(t.get("delta_sentiment", 0.0)))
            # Placeholder: no true oracle available in simulator API.
            immediate_regret.append(float(t.get("immediate_regret", 0.0)))
        episode_rewards.append(ep_reward)

    outcomes = summarize_episode_outcomes(episode_logs)
    n_episodes = max(len(episode_logs), 1)
    return {
        "avg_per_turn_reward": float(np.mean(all_turn_rewards)) if all_turn_rewards else 0.0,
        "early_turn_reward": float(np.mean(early_turn_rewards)) if early_turn_rewards else 0.0,
        "sentiment_improvement": float(np.sum(all_sentiment_delta)),
        "immediate_regret": float(np.mean(immediate_regret)) if immediate_regret else 0.0,
        "resolution_rate": float(outcomes["success"] / n_episodes),
        "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
    }


def compute_rl_metrics(episode_logs: list[dict[str, Any]]) -> dict[str, float]:
    if not episode_logs:
        return {
            "cumulative_episode_reward": 0.0,
            "resolution_rate": 0.0,
            "mean_turns": 0.0,
        }
    rewards = [float(ep.get("episode_reward", 0.0)) for ep in episode_logs]
    turns = [int(ep.get("turn_count", 0)) for ep in episode_logs]
    outcomes = summarize_episode_outcomes(episode_logs)
    n_episodes = max(len(episode_logs), 1)
    return {
        "cumulative_episode_reward": float(np.mean(rewards)) if rewards else 0.0,
        "resolution_rate": float(outcomes["success"] / n_episodes),
        "mean_turns": float(np.mean(turns)) if turns else 0.0,
    }
