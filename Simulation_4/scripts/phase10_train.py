from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Simulation_4.training.train_ppo import train_ppo


"""
Phase 10: PPO Training
Usage: python Simulation_4/scripts/phase10_train.py [--no-curriculum] [--no-shaping] [--timesteps N]
"""


def train(args: argparse.Namespace) -> dict:
    artifacts_root = Path("Simulation_4/artifacts")
    phase10_root = artifacts_root / "phase10_v3"
    phase10_root.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  PHASE 10: PPO TRAINING")
    print("=" * 65)
    print(f"  Timesteps:    {args.timesteps:,}")
    print(f"  Curriculum:   {'enabled' if not args.no_curriculum else 'disabled'}")
    print("  Reward shape: disabled (forced for v3)")
    print("  Parallel envs: 4")
    print("=" * 65)

    summary = train_ppo(
        artifacts_root=str(artifacts_root),
        timesteps=int(args.timesteps),
        use_curriculum=not bool(args.no_curriculum),
        use_reward_shaping=False,
        output_subdir="phase10_v3",
        n_envs=4,
        seed=42,
    )

    print("\nTraining summary:")
    print(json.dumps(summary, indent=2))
    print("\nArtifacts:")
    print("  - Simulation_4/artifacts/phase10_v3/models/best_model.zip")
    print("  - Simulation_4/artifacts/phase10_v3/models/final_model.zip")
    print("  - Simulation_4/artifacts/phase10_v3/tensorboard/")
    print("  - Simulation_4/artifacts/phase10_v3/training_log.json")
    print("  - Simulation_4/artifacts/phase10_v3/training_summary.json")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--no-shaping", action="store_true")
    parser.add_argument("--timesteps", type=int, default=500_000)
    train(parser.parse_args())
