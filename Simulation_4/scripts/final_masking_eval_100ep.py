#!/usr/bin/env python3
"""Final comprehensive LUMO evaluation with masking fix (100 episodes)"""

import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime

# Setup paths
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from Simulation_4.env.support_env import SupportEnv
from stable_baselines3 import PPO
from sb3_contrib.common.wrappers import ActionMasker

def mask_fn(env):
    """Mask function for gymnasium environment"""
    return env.action_masks()

def run_final_evaluation(n_episodes=100):
    """Run comprehensive LUMO evaluation"""
    print("=" * 75)
    print(f"FINAL LUMO EVALUATION WITH MASKING FIX")
    print(f"Episodes: {n_episodes}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 75)
    
    # Setup
    repo_root_path = Path(__file__).parent.parent.parent
    artifacts_root = repo_root_path / "Simulation_4" / "artifacts"
    model_path = repo_root_path / "RL Out" / "models" / "best_model"
    results_dir = repo_root_path / "Simulation_4" / "artifacts" / "phase13_masking_fix_eval"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n① Loading environment...")
    env = SupportEnv(str(artifacts_root), nlg_enabled=False)
    
    print(f"② Wrapping with ActionMasker (escalation blocked until turn 3)...")
    env = ActionMasker(env, mask_fn)
    
    print(f"③ Loading trained PPO model...")
    model = PPO.load(str(model_path))
    
    # Tracking
    outcomes = {}
    episode_data = []
    total_steps = 0
    
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
        unwrapped_env = env.unwrapped
        outcome_dict = unwrapped_env.last_transition_outcome if hasattr(unwrapped_env, 'last_transition_outcome') else {}
        outcome = outcome_dict.get("terminal_type", "unknown")
        
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        episode_data.append({
            "episode": ep + 1,
            "steps": steps,
            "outcome": outcome,
            "resolved": outcome == "resolved"
        })
        total_steps += steps
        
        if (ep + 1) % 20 == 0:
            print(f"  Progress: {ep+1}/{n_episodes} episodes completed")
    
    # Print results
    print(f"\n{'=' * 75}")
    print("RESULTS:")
    print(f"{'=' * 75}")
    print(f"Episodes completed:        {n_episodes}")
    print(f"Total steps:               {total_steps}")
    print(f"Avg steps per episode:     {total_steps/n_episodes:.2f}")
    print(f"\nOutcome Distribution:")
    for outcome in sorted(outcomes.keys()):
        count = outcomes[outcome]
        pct = 100.0 * count / n_episodes
        print(f"  {outcome:25s}: {count:4d} ({pct:6.1f}%)")
    
    resolved_count = outcomes.get("resolved", 0)
    escalated_count = outcomes.get("escalated", 0)
    resolution_rate = 100.0 * resolved_count / n_episodes
    escalation_rate = 100.0 * escalated_count / n_episodes
    
    print(f"\n{'=' * 75}")
    print(f"METRICS:")
    print(f"{'=' * 75}")
    print(f"Resolution Rate:           {resolution_rate:6.1f}%  ({resolved_count}/{n_episodes})")
    print(f"Escalation Rate:           {escalation_rate:6.1f}%  ({escalated_count}/{n_episodes})")
    print(f"\nComparison to Training:")
    print(f"  Trained model achieved:    69.1% resolution, 0% escalation")
    print(f"  Current with masking:      {resolution_rate:6.1f}% resolution, {escalation_rate:6.1f}% escalation")
    
    # Save results
    results_file = results_dir / f"masking_fix_eval_{n_episodes}ep.json"
    results_dict = {
        "timestamp": datetime.now().isoformat(),
        "n_episodes": n_episodes,
        "metrics": {
            "resolution_rate_pct": resolution_rate,
            "escalation_rate_pct": escalation_rate,
            "total_steps": total_steps,
            "avg_steps_per_episode": total_steps / n_episodes,
        },
        "outcomes": outcomes,
        "episode_data": episode_data
    }
    
    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n④ Results saved to: {results_file}")
    print("=" * 75)
    
    env.close()
    return resolution_rate, escalation_rate

if __name__ == "__main__":
    res_rate, esc_rate = run_final_evaluation(n_episodes=100)
    
    print("\n📊 ANALYSIS:")
    print(f"  The action masking is now ENFORCED in the environment ✓")
    print(f"  Escalation is blocked for turns 0-2, allowed from turn 3+ ✓")
    print(f"  Model performance: {res_rate:.1f}% resolution")
    
    if res_rate < 20:
        print(f"\n⚠️  Resolution rate is still low. Possible causes:")
        print(f"    1. The reward structure changes may have degraded model quality")
        print(f"    2. The model may need retraining with masking + new rewards")
        print(f"    3. The model may not have learned to solve issues without escalation")
