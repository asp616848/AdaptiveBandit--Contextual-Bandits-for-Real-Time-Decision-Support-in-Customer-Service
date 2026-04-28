from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from simulation.training.train_ppo import continue_ppo_from_checkpoint


"""
Phase 10: Continue PPO training from v3 checkpoint.
Usage: python simulation/scripts/phase10_continue_train.py [--timesteps N] [--checkpoint PATH]
"""


def train(args: argparse.Namespace) -> dict:
    artifacts_root = Path("simulation/artifacts")
    phase_root = artifacts_root / "phase10_v3_continued"
    phase_root.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  PHASE 10: PPO CONTINUED TRAINING (v3 -> v3_continued)")
    print("=" * 65)
    print(f"  Additional timesteps: {args.timesteps:,}")
    print(f"  Checkpoint:           {args.checkpoint}")
    print("  Curriculum:           enabled")
    print("  Reward shaping:       disabled")
    print("  Parallel envs:        4")
    print("  reset_num_timesteps:  False")
    print("=" * 65)

    summary = continue_ppo_from_checkpoint(
        artifacts_root=str(artifacts_root),
        checkpoint_path=str(args.checkpoint),
        additional_timesteps=int(args.timesteps),
        use_curriculum=True,
        use_reward_shaping=False,
        output_subdir="phase10_v3_continued",
        n_envs=4,
        seed=42,
        tb_log_name="ppo_v3_continued",
    )

    print("\nTraining summary:")
    print(json.dumps(summary, indent=2))
    print("\nArtifacts:")
    print("  - simulation/artifacts/phase10_v3_continued/models/best_model.zip")
    print("  - simulation/artifacts/phase10_v3_continued/models/final_model.zip")
    print("  - simulation/artifacts/phase10_v3_continued/tensorboard/")
    print("  - simulation/artifacts/phase10_v3_continued/training_log.json")
    print("  - simulation/artifacts/phase10_v3_continued/training_summary.json")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="simulation/artifacts/phase10_final/models/best_model.zip",
    )
    train(parser.parse_args())
