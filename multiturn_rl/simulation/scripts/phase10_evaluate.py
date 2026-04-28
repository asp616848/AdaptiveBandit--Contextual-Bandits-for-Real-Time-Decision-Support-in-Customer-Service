from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable
import sys

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
from stable_baselines3 import PPO

from simulation.env.support_env import SupportEnv
from simulation.validation.baseline_policies import (
    policy_always_escalate,
    policy_always_solve,
    policy_document_guided,
    policy_random,
    policy_threshold_escalate,
)


def evaluate_model(model_path: str, artifacts_root: str, n_episodes: int = 1000) -> dict[str, Any]:
    model = PPO.load(model_path)
    env = SupportEnv(artifacts_root=artifacts_root, nlg_enabled=False)

    results: dict[str, Any] = {
        "rewards": [],
        "resolution_count": 0,
        "escalation_count": 0,
        "dropout_count": 0,
        "timeout_count": 0,
        "turns_to_resolution": [],
        "escalation_by_tier": {"Free": 0, "Pro": 0, "Business": 0, "Enterprise": 0},
        "resolution_by_persona": {
            p: 0
            for p in [
                "high_engagement_resolver",
                "low_engagement_resolver",
                "silent_dropout",
                "escalation_prone",
            ]
        },
        "persona_counts": {
            p: 0
            for p in [
                "high_engagement_resolver",
                "low_engagement_resolver",
                "silent_dropout",
                "escalation_prone",
            ]
        },
    }

    for ep in range(int(n_episodes)):
        obs, _ = env.reset(seed=50_000 + ep)
        done = False
        ep_reward = 0.0
        info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            ep_reward += float(reward)
            if truncated:
                break

        results["rewards"].append(ep_reward)

        terminal = str((info.get("last_transition_outcome", {}) or {}).get("terminal_type", "timeout"))
        tier = str(env.state.get("tier", "Free"))
        persona = str(env.state.get("persona_label", ""))

        if terminal == "success":
            results["resolution_count"] += 1
            results["turns_to_resolution"].append(int(env.state.get("turn_count", 0)))
            if persona in results["resolution_by_persona"]:
                results["resolution_by_persona"][persona] += 1
        elif terminal == "escalation":
            results["escalation_count"] += 1
            if tier in results["escalation_by_tier"]:
                results["escalation_by_tier"][tier] += 1
        elif terminal == "dropout":
            results["dropout_count"] += 1
        else:
            results["timeout_count"] += 1

        if persona in results["persona_counts"]:
            results["persona_counts"][persona] += 1

    total = max(int(n_episodes), 1)
    results["mean_reward"] = float(np.mean(results["rewards"])) if results["rewards"] else 0.0
    results["std_reward"] = float(np.std(results["rewards"])) if results["rewards"] else 0.0
    results["resolution_rate"] = float(results["resolution_count"] / total)
    results["escalation_rate"] = float(results["escalation_count"] / total)
    results["dropout_rate"] = float(results["dropout_count"] / total)
    results["timeout_rate"] = float(results["timeout_count"] / total)
    results["mean_turns_to_resolution"] = (
        float(np.mean(results["turns_to_resolution"])) if results["turns_to_resolution"] else 0.0
    )
    env.close()
    return results


def _call_policy(policy_fn: Callable, obs: dict[str, float], state: dict[str, Any], rng: np.random.Generator) -> int:
    try:
        return int(policy_fn(obs, state, rng))
    except TypeError:
        return int(policy_fn(obs, state))


def run_baseline_episodes(
    policy_fn: Callable,
    artifacts_root: str,
    n_episodes: int,
    seed_offset: int = 60_000,
) -> dict[str, Any]:
    rng = np.random.default_rng(123)
    env = SupportEnv(artifacts_root=artifacts_root, nlg_enabled=False)

    rewards = []
    resolution_count = 0
    escalation_count = 0
    dropout_count = 0
    timeout_count = 0
    turns_to_resolution = []

    for ep in range(int(n_episodes)):
        obs, _ = env.reset(seed=int(seed_offset + ep))
        done = False
        info = {}
        ep_reward = 0.0
        while not done:
            obs_dict = {
                "information": float(env.state.get("information", 0.0)),
                "progress": float(env.state.get("progress", 0.0)),
                "frustration": float(env.state.get("frustration", 0.0)),
                "failed_streak": int(env.state.get("failed_streak", 0)),
            }
            action = _call_policy(policy_fn, obs_dict, dict(env.state), rng)
            obs, reward, done, truncated, info = env.step(int(action))
            ep_reward += float(reward)
            if truncated:
                break

        rewards.append(ep_reward)
        terminal = str((info.get("last_transition_outcome", {}) or {}).get("terminal_type", "timeout"))
        if terminal == "success":
            resolution_count += 1
            turns_to_resolution.append(int(env.state.get("turn_count", 0)))
        elif terminal == "escalation":
            escalation_count += 1
        elif terminal == "dropout":
            dropout_count += 1
        else:
            timeout_count += 1

    env.close()

    total = max(int(n_episodes), 1)
    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "resolution_rate": float(resolution_count / total),
        "escalation_rate": float(escalation_count / total),
        "dropout_rate": float(dropout_count / total),
        "timeout_rate": float(timeout_count / total),
        "mean_turns_to_resolution": float(np.mean(turns_to_resolution)) if turns_to_resolution else 0.0,
    }


def compare_all_policies(artifacts_root: str, n_episodes: int = 1000, model_name: str = "best_model") -> dict[str, Any]:
    model_path = str(Path("simulation/artifacts/phase10/models") / model_name)
    ppo_results = evaluate_model(model_path, artifacts_root, n_episodes)

    baselines = {
        "document_guided": policy_document_guided,
        "threshold_escalate": policy_threshold_escalate,
        "random": policy_random,
        "always_solve": policy_always_solve,
        "always_escalate": policy_always_escalate,
    }

    baseline_results: dict[str, dict[str, Any]] = {}
    for name, policy_fn in baselines.items():
        baseline_results[name] = run_baseline_episodes(
            policy_fn,
            artifacts_root,
            n_episodes,
            seed_offset=60_000,
        )

    print("\n" + "=" * 75)
    print("  POLICY COMPARISON - PPO vs Baselines")
    print("=" * 75)
    print(f"  {'Policy':<25} {'Mean Reward':>12} {'Resolution':>12} {'Escalation':>12} {'Turns':>8}")
    print("-" * 75)

    print(
        f"  {'PPO (trained)':<25} "
        f"{ppo_results['mean_reward']:>12.3f} "
        f"{ppo_results['resolution_rate']:>11.1%} "
        f"{ppo_results['escalation_rate']:>11.1%} "
        f"{ppo_results['mean_turns_to_resolution']:>8.1f}"
    )

    for name, res in sorted(baseline_results.items(), key=lambda x: x[1]["mean_reward"], reverse=True):
        print(
            f"  {name:<25} "
            f"{res['mean_reward']:>12.3f} "
            f"{res['resolution_rate']:>11.1%} "
            f"{res['escalation_rate']:>11.1%} "
            f"{res['mean_turns_to_resolution']:>8.1f}"
        )

    print("=" * 75)
    print("\n  Escalation by tier (PPO):")
    for tier, count in ppo_results["escalation_by_tier"].items():
        rate = float(count) / max(float(n_episodes), 1.0)
        print(f"    {tier}: {rate:.1%}")

    print("\n  Expected: Enterprise > Business > Pro > Free escalation rate")

    return {
        "ppo": ppo_results,
        "baselines": baseline_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="best_model")
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()

    artifacts_root = "simulation/artifacts"
    report = compare_all_policies(artifacts_root, n_episodes=args.episodes, model_name=args.model)

    out_path = Path("simulation/artifacts/phase10/evaluation_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation report: {out_path}")
