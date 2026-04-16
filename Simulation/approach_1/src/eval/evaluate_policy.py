from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from src.env.simple_support_env import SimpleSupportEnv


def run_policy(env: SimpleSupportEnv, model: PPO | None, episodes: int, random_policy: bool) -> dict:
    total_reward = 0.0
    success = 0
    escalate = 0
    close_early = 0
    timeout = 0
    completed = 0

    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        terminal_reason = None

        while not done:
            if random_policy:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, step_info = env.step(int(action))
            ep_reward += float(reward)
            terminal_reason = step_info.get("terminal_reason")

        total_reward += ep_reward
        if terminal_reason == "completed":
            success += 1
            completed += 1
        elif terminal_reason == "escalate":
            escalate += 1
        elif terminal_reason == "close_early":
            close_early += 1
        elif terminal_reason == "timeout":
            timeout += 1

    return {
        "episodes": episodes,
        "avg_reward": total_reward / max(episodes, 1),
        "success_rate": success / max(episodes, 1),
        "completed_count": completed,
        "escalate_rate": escalate / max(episodes, 1),
        "close_early_rate": close_early / max(episodes, 1),
        "timeout_rate": timeout / max(episodes, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained policy against random baseline.")
    parser.add_argument("--catalog", default="Simulation_4/approach_1/data/processed/subflow_catalog.json")
    parser.add_argument("--reward-config", default="Simulation_4/approach_1/configs/reward_config.json")
    parser.add_argument("--model", default="Simulation_4/approach_1/data/models/ppo_simple_final.zip")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--out", default="Simulation_4/approach_1/data/reports/eval_report.json")
    args = parser.parse_args()

    env = SimpleSupportEnv.from_files(args.catalog, args.reward_config, seed=42)
    model = PPO.load(args.model)

    trained_metrics = run_policy(env, model=model, episodes=args.episodes, random_policy=False)

    random_env = SimpleSupportEnv.from_files(args.catalog, args.reward_config, seed=123)
    random_metrics = run_policy(random_env, model=None, episodes=args.episodes, random_policy=True)

    report = {
        "trained": trained_metrics,
        "random": random_metrics,
        "improvement_success_rate": trained_metrics["success_rate"] - random_metrics["success_rate"],
        "improvement_avg_reward": trained_metrics["avg_reward"] - random_metrics["avg_reward"],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
