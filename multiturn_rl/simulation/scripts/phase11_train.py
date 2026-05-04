from __future__ import annotations

"""
Phase 11: PPO Training — Stage 1 Improvements
=============================================
Fixes applied vs phase10_textonly_persist:
  - Uses 9D state observation (not HashingVectorizer text-only)
  - NLG enabled: agent and customer exchange real language, conversation
    history is built up and fed back into each NLG call
  - Reward shaping enabled (potential-based, policy-invariant)
  - Escalation costs raised: Free 4.0→6.0, Pro 2.0→3.0
  - Escalate action masked for turn_count < 3
  - 500k timesteps with full curriculum

Usage:
  python simulation/scripts/phase11_train.py [--timesteps N] [--no-curriculum] [--no-shaping]
"""

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from simulation.training.train_ppo import train_ppo


def train(args: argparse.Namespace) -> dict:
    artifacts_root = Path("simulation/artifacts")
    phase_root = artifacts_root / "phase11"
    phase_root.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  PHASE 11: PPO TRAINING — STAGE 1 IMPROVEMENTS")
    print("=" * 65)
    print(f"  Timesteps:      {args.timesteps:,}")
    print(f"  Curriculum:     {'enabled' if not args.no_curriculum else 'disabled'}")
    print(f"  Reward shaping: {'enabled' if not args.no_shaping else 'disabled'}")
    print("  Observation:    9D state (no text wrapper)")
    print("  NLG:            enabled — real language conversation")
    print("  Escalation cost: Free=6.0, Pro=3.0, Business=0.5, Enterprise=0.0")
    print("  Action mask:    Escalate blocked for turn_count < 3")
    print("  Parallel envs:  4")
    print("=" * 65)

    summary = train_ppo(
        artifacts_root=str(artifacts_root),
        timesteps=int(args.timesteps),
        use_curriculum=not bool(args.no_curriculum),
        use_reward_shaping=not bool(args.no_shaping),
        output_subdir="phase11",
        n_envs=4,
        seed=42,
        nlg_enabled=True,
    )

    print("\nTraining summary:")
    print(json.dumps(summary, indent=2))
    print("\nArtifacts:")
    print("  - simulation/artifacts/phase11/models/best_model.zip")
    print("  - simulation/artifacts/phase11/models/final_model.zip")
    print("  - simulation/artifacts/phase11/tensorboard/")
    print("  - simulation/artifacts/phase11/training_log.json")
    print("  - simulation/artifacts/phase11/training_summary.json")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 11 PPO training — Stage 1 improvements")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total training timesteps")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    parser.add_argument("--no-shaping", action="store_true", help="Disable reward shaping")
    train(parser.parse_args())
