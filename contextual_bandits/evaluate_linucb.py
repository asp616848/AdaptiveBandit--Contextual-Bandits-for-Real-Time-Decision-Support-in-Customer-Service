#!/usr/bin/env python3
"""
Evaluate and Compare Per-Turn LinUCB vs PPO on SupportEnv.

Shows the performance difference between:
- LinUCB (treats each turn independently)
- PPO (learns full MDP with multi-step strategy)
"""

import argparse
import sys
from pathlib import Path
import json
import time
import numpy as np

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Simulation_4.env.support_env import SupportEnv
from Simulation_4.contextual_bandits.linucb import PerTurnLinUCBPolicy
from stable_baselines3 import PPO


def evaluate_policy(
    env: SupportEnv,
    policy,
    policy_type: str,
    n_episodes: int = 100,
) -> dict:
    """
    Evaluate a policy (LinUCB or PPO).
    
    Args:
        env: Environment
        policy: Trained policy (PerTurnLinUCBPolicy or PPO)
        policy_type: "linucb" or "ppo"
        n_episodes: Number of episodes
    
    Returns:
        Evaluation metrics
    """
    outcomes = {"resolved": 0, "escalated": 0, "dropout": 0, "timeout": 0, "other": 0}
    rewards = []
    turns_list = []
    
    print(f"\nEvaluating {policy_type.upper()} on {n_episodes} episodes...")
    
    for ep in range(n_episodes):
        if (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"  Progress: {ep + 1}/{n_episodes}")
        
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        n_turns = 0
        
        while not done:
            turn_count = int(env.state.get("turn_count", 0))
            
            if policy_type == "linucb":
                action = policy.predict_action(obs, turn_count)
            else:  # ppo
                action, _ = policy.predict(obs, deterministic=True)
            
            obs, reward, done, truncated, info = env.step(int(action))
            ep_reward += float(reward)
            n_turns += 1
            
            if truncated:
                break
        
        terminal_type = info.get("last_transition_outcome", {}).get("terminal_type", "other")
        if terminal_type in outcomes:
            outcomes[terminal_type] += 1
        else:
            outcomes["other"] += 1
        
        rewards.append(ep_reward)
        turns_list.append(n_turns)
    
    # Calculate metrics
    resolution_rate = 100.0 * outcomes.get("resolved", 0) / n_episodes if n_episodes > 0 else 0
    escalation_rate = 100.0 * outcomes.get("escalated", 0) / n_episodes if n_episodes > 0 else 0
    
    return {
        "policy_type": policy_type,
        "n_episodes": n_episodes,
        "outcomes": outcomes,
        "resolution_rate": resolution_rate,
        "escalation_rate": escalation_rate,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_turns": float(np.mean(turns_list)),
        "std_turns": float(np.std(turns_list)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare Per-Turn LinUCB vs PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--linucb-model-dir",
        type=str,
        default="Simulation_4/artifacts/linucb_models/models",
        help="Directory with trained LinUCB models",
    )
    parser.add_argument(
        "--ppo-model",
        type=str,
        default="RL Out/models/best_model",
        help="Path to trained PPO model",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes for comparison",
    )
    parser.add_argument(
        "--artifacts-root",
        type=str,
        default="Simulation_4/artifacts",
        help="Artifacts root for environment",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Simulation_4/artifacts/linucb_vs_ppo_comparison.json",
        help="Output file for comparison results",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LinUCB vs PPO COMPARISON")
    print("=" * 70)
    
    # Initialize environment
    print(f"\n① Initializing environment...")
    env = SupportEnv(artifacts_root=args.artifacts_root, nlg_enabled=False)
    print(f"✓ Environment ready")
    
    results_all = {}
    
    # Load and evaluate LinUCB
    print(f"\n② Loading LinUCB models from {args.linucb_model_dir}")
    try:
        linucb_policy = PerTurnLinUCBPolicy.load(args.linucb_model_dir)
        print(f"✓ LinUCB models loaded")
        
        linucb_results = evaluate_policy(
            env, linucb_policy, "linucb", n_episodes=args.episodes
        )
        results_all["linucb"] = linucb_results
        print(f"✓ LinUCB evaluation complete")
    except Exception as e:
        print(f"✗ Failed to load LinUCB: {e}")
        linucb_results = None
    
    # Load and evaluate PPO
    print(f"\n③ Loading PPO model from {args.ppo_model}")
    try:
        ppo_model = PPO.load(args.ppo_model)
        print(f"✓ PPO model loaded")
        
        ppo_results = evaluate_policy(
            env, ppo_model, "ppo", n_episodes=args.episodes
        )
        results_all["ppo"] = ppo_results
        print(f"✓ PPO evaluation complete")
    except Exception as e:
        print(f"✗ Failed to load PPO: {e}")
        ppo_results = None
    
    # Print comparison
    print(f"\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    if linucb_results and ppo_results:
        print(f"\n{'Metric':<30} {'LinUCB':>15} {'PPO':>15} {'Difference':>15}")
        print("-" * 75)
        
        metrics = [
            ("Resolution Rate", "resolution_rate"),
            ("Escalation Rate", "escalation_rate"),
            ("Mean Reward", "mean_reward"),
            ("Mean Turns", "mean_turns"),
        ]
        
        for metric_name, metric_key in metrics:
            linucb_val = linucb_results[metric_key]
            ppo_val = ppo_results[metric_key]
            diff = linucb_val - ppo_val
            
            print(f"{metric_name:<30} {linucb_val:>14.1f}% {ppo_val:>14.1f}% {diff:>14.1f}%")
        
        print(f"\n{'Outcome Distribution':<30} {'LinUCB':>15} {'PPO':>15}")
        print("-" * 60)
        
        for outcome in ["resolved", "escalated", "dropout"]:
            linucb_count = linucb_results["outcomes"].get(outcome, 0)
            ppo_count = ppo_results["outcomes"].get(outcome, 0)
            linucb_pct = 100.0 * linucb_count / args.episodes
            ppo_pct = 100.0 * ppo_count / args.episodes
            
            print(f"{outcome.capitalize():<30} {linucb_pct:>14.1f}% {ppo_pct:>14.1f}%")
    
    elif linucb_results:
        print("\n✓ LinUCB Results:")
        print(f"  Resolution: {linucb_results['resolution_rate']:.1f}%")
        print(f"  Escalation: {linucb_results['escalation_rate']:.1f}%")
    
    elif ppo_results:
        print("\n✓ PPO Results:")
        print(f"  Resolution: {ppo_results['resolution_rate']:.1f}%")
        print(f"  Escalation: {ppo_results['escalation_rate']:.1f}%")
    
    # Interpretation
    print(f"\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(f"""
The per-turn LinUCB model treats each turn as an independent decision problem.
It CANNOT learn multi-step strategies like:
  - Turn 0: Gather information
  - Turn 1: Provide solution
  - Turn 2: Resolve issue

PPO learns the full MDP and can discover these sequential patterns.

Expected result: PPO >> LinUCB in resolution rate
Reason: LinUCB's per-turn independence ignores state outcomes that matter for
        future decisions.
""")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results_all, f, indent=2)
    
    print(f"✓ Comparison results saved to {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
