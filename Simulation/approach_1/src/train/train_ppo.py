from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from src.env.simple_support_env import SimpleSupportEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on Approach 1 simple environment.")
    parser.add_argument("--catalog", default="Simulation_4/approach_1/data/processed/subflow_catalog.json")
    parser.add_argument("--reward-config", default="Simulation_4/approach_1/configs/reward_config.json")
    parser.add_argument("--train-config", default="Simulation_4/approach_1/configs/train_config.json")
    parser.add_argument("--out-dir", default="Simulation_4/approach_1/data/models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_cfg = json.loads(Path(args.train_config).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def env_fn():
        return SimpleSupportEnv.from_files(
            catalog_path=args.catalog,
            reward_config_path=args.reward_config,
            seed=args.seed,
        )

    n_envs = int(train_cfg.get("n_envs", 4))
    vec_env = make_vec_env(env_fn, n_envs=n_envs, seed=args.seed)

    model = PPO(
        policy=train_cfg.get("policy", "MlpPolicy"),
        env=vec_env,
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        n_steps=int(train_cfg.get("n_steps", 1024)),
        batch_size=int(train_cfg.get("batch_size", 64)),
        gamma=float(train_cfg.get("gamma", 0.99)),
        gae_lambda=float(train_cfg.get("gae_lambda", 0.95)),
        clip_range=float(train_cfg.get("clip_range", 0.2)),
        ent_coef=float(train_cfg.get("ent_coef", 0.0)),
        vf_coef=float(train_cfg.get("vf_coef", 0.5)),
        seed=args.seed,
        verbose=1,
    )

    total_timesteps = int(train_cfg.get("total_timesteps", 200000))
    model.learn(total_timesteps=total_timesteps)

    final_path = out_dir / "ppo_simple_final"
    model.save(str(final_path))

    eval_env = env_fn()
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=int(train_cfg.get("eval_episodes", 100)),
        deterministic=True,
    )

    metrics = {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "total_timesteps": total_timesteps,
        "model_path": str(final_path),
    }

    metrics_path = out_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
