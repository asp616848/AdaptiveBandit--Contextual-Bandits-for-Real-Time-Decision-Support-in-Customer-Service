#!/usr/bin/env python3
"""LUMO evaluation with proper stable-baselines3 action masking"""

import sys
from pathlib import Path
import numpy as np

# Setup paths
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from Simulation_4.env.support_env import SupportEnv
from stable_baselines3 import PPO
from sb3_contrib.common.wrappers import ActionMasker

def mask_fn(env):
    """Mask function for gymnasium environment"""
    return env.action_masks()

def run_lumo_with_masker(n_episodes=20):
    """Run LUMO evaluation with proper action masking wrapper"""
    print("=" * 70)
    print(f"LUMO EVALUATION (proper masking wrapper) - {n_episodes} episodes")
    print("=" * 70)
    
    # Setup
    repo_root_path = Path(__file__).parent.parent.parent
    artifacts_root = repo_root_path / "Simulation_4" / "artifacts"
    model_path = repo_root_path / "RL Out" / "models" / "best_model"
    
    print(f"\n① Loading environment from: {artifacts_root}")
    env = SupportEnv(str(artifacts_root), nlg_enabled=False)
    
    # Wrap with ActionMasker for proper masking
    print(f"② Wrapping environment with ActionMasker")
    env = ActionMasker(env, mask_fn)
    
    print(f"③ Loading model from: {model_path}")
    model = PPO.load(str(model_path))
    
    # Tracking
    outcomes = {
        "resolved": 0,
        "escalated": 0,
        "timeout": 0,
        "unresolved_close": 0,
    }
    total_steps = 0
    episodes_completed = 0
    
    print(f"\n④ Running {n_episodes} episodes with deterministic policy...\n")
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done and steps < env.unwrapped.T_max:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            steps += 1
        
        # Record outcome
        outcome = env.unwrapped.last_transition_outcome.get("terminal_type", "timeout") if hasattr(env.unwrapped, 'last_transition_outcome') else "timeout"
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        total_steps += steps
        episodes_completed += 1
        
        status = "RESOLVED" if outcome == "resolved" else outcome.upper()
        print(f"Episode {ep+1:2d}/{n_episodes}: {status:20s} (steps={steps:2d})")
    
    # Print results
    print(f"\n{'=' * 70}")
    print("RESULTS:")
    print(f"{'=' * 70}")
    print(f"Episodes completed: {episodes_completed}")
    print(f"Total steps: {total_steps}")
    print(f"Avg steps per episode: {total_steps/episodes_completed:.1f}")
    print(f"\nOutcome Distribution:")
    for outcome, count in outcomes.items():
        pct = 100.0 * count / episodes_completed if episodes_completed > 0 else 0
        print(f"  {outcome:20s}: {count:3d} ({pct:5.1f}%)")
    
    resolution_rate = 100.0 * outcomes.get("resolved", 0) / episodes_completed if episodes_completed > 0 else 0
    escalation_rate = 100.0 * outcomes.get("escalated", 0) / episodes_completed if episodes_completed > 0 else 0
    
    print(f"\n{'=' * 70}")
    print(f"Resolution Rate: {resolution_rate:.1f}%")
    print(f"Escalation Rate: {escalation_rate:.1f}%")
    print(f"{'=' * 70}\n")
    
    env.close()
    return resolution_rate, escalation_rate

if __name__ == "__main__":
    try:
        res_rate, esc_rate = run_lumo_with_masker(n_episodes=20)
        
        # Expected vs Actual
        print("BENCHMARK COMPARISON:")
        print(f"  Training expected:    ~69.1% resolution, ~0% escalation")
        print(f"  Current evaluation:    {res_rate:.1f}% resolution, {esc_rate:.1f}% escalation")
        print()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nNote: If sb3-contrib is not available, you may need to install it:")
        print("  pip install sb3-contrib")
