#!/usr/bin/env python3
"""Quick LUMO evaluation test with fixed masking"""

import sys
from pathlib import Path
import numpy as np

# Setup paths
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from Simulation_4.env.support_env import SupportEnv
from Simulation_4.rag.lumo_rag import LumoRAG
from stable_baselines3 import PPO

def run_quick_evaluation(n_episodes=10):
    """Run quick LUMO evaluation with masking fix"""
    print("=" * 70)
    print(f"QUICK LUMO EVALUATION (masking-fixed) - {n_episodes} episodes")
    print("=" * 70)
    
    # Setup
    repo_root_path = Path(__file__).parent.parent.parent
    artifacts_root = repo_root_path / "Simulation_4" / "artifacts"
    model_path = repo_root_path / "RL Out" / "models" / "best_model"
    
    print(f"\n① Loading environment from: {artifacts_root}")
    env = SupportEnv(str(artifacts_root), nlg_enabled=False)
    
    print(f"② Loading model from: {model_path}")
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
    action_overrides = 0
    
    print(f"\n③ Running {n_episodes} episodes with deterministic policy...\n")
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        ep_masked_actions = 0
        
        while not done and steps < env.T_max:
            # Get action from policy with masking
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            
            # Check if action is masked
            masks = env.action_masks()
            if not masks[action]:
                # This will be overridden in step()
                ep_masked_actions += 1
            
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
        
        # Record outcome
        outcome = info.get("terminal_type", "timeout")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        total_steps += steps
        episodes_completed += 1
        action_overrides += ep_masked_actions
        
        status = "RESOLVED" if outcome == "resolved" else outcome.upper()
        print(f"Episode {ep+1:2d}/{n_episodes}: {status:20s} (steps={steps:2d}, masked_actions={ep_masked_actions})")
    
    # Print results
    print(f"\n{'=' * 70}")
    print("RESULTS:")
    print(f"{'=' * 70}")
    print(f"Episodes completed: {episodes_completed}")
    print(f"Total steps: {total_steps}")
    print(f"Avg steps per episode: {total_steps/episodes_completed:.1f}")
    print(f"Total action overrides (masking applied): {action_overrides}")
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
    
    return resolution_rate, escalation_rate

if __name__ == "__main__":
    res_rate, esc_rate = run_quick_evaluation(n_episodes=10)
    
    # Expected vs Actual
    print("COMPARISON:")
    print(f"  Training achieved:      69.1% resolution, 0% escalation")
    print(f"  Current evaluation:     {res_rate:.1f}% resolution, {esc_rate:.1f}% escalation")
    if res_rate > 50:
        print("  ✓ MASKING FIX WORKING!")
    else:
        print("  ✗ Still problematic - may need further investigation")
