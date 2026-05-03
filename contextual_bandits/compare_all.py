#!/usr/bin/env python3
"""
Compare Three Approaches to Customer Support Optimization:
1. PPO (Multi-turn adaptive)
2. Strategy LinUCB (Single-turn fixed strategy)
3. Per-Turn LinUCB (Per-turn independent decisions)
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Simulation_4.env.support_env import SupportEnv
from Simulation_4.contextual_bandits.env_factory import build_bandit_env
from Simulation_4.contextual_bandits.strategy_linucb import StrategyLinUCB, StrategyExecutor, STRATEGIES
from Simulation_4.contextual_bandits.linucb import PerTurnLinUCBPolicy
from stable_baselines3 import PPO


def evaluate_ppo(env: SupportEnv, model: PPO, n_episodes: int = 100) -> dict:
    """Evaluate PPO policy (adaptive multi-turn)."""
    print(f"\nEvaluating PPO (adaptive multi-turn policy)...")
    
    outcomes = {"resolved": 0, "escalated": 0, "dropout": 0, "timeout": 0, "other": 0}
    turns_list = []
    
    for ep in range(n_episodes):
        if (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"  {ep + 1}/{n_episodes}")
        
        obs, _ = env.reset()
        done = False
        turn = 0
        
        while not done and turn < env.T_max:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            turn += 1
            
            if truncated:
                break
        
        terminal_type = info.get("last_transition_outcome", {}).get("terminal_type", "unknown")
        if terminal_type in outcomes:
            outcomes[terminal_type] += 1
        else:
            outcomes["other"] += 1
        
        turns_list.append(turn)
    
    resolution_rate = 100.0 * outcomes.get("resolved", 0) / n_episodes if n_episodes > 0 else 0
    escalation_rate = 100.0 * outcomes.get("escalated", 0) / n_episodes if n_episodes > 0 else 0
    
    return {
        "policy_type": "PPO (Multi-turn Adaptive)",
        "outcomes": outcomes,
        "resolution_rate": resolution_rate,
        "escalation_rate": escalation_rate,
        "mean_turns": float(np.mean(turns_list)) if turns_list else 0,
    }


def evaluate_strategy_linucb(env: SupportEnv, policy: StrategyLinUCB, n_episodes: int = 100) -> dict:
    """Evaluate Strategy LinUCB (single fixed strategy per conversation)."""
    print(f"\nEvaluating Strategy LinUCB (fixed strategy selection)...")
    
    outcomes = {"resolved": 0, "escalated": 0, "dropout": 0, "timeout": 0, "other": 0}
    turns_list = []
    strategy_dist = np.zeros(3)
    
    for ep in range(n_episodes):
        if (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"  {ep + 1}/{n_episodes}")
        
        obs, _ = env.reset()
        initial_obs = obs.copy()
        
        strategy = policy.predict(initial_obs)
        strategy_dist[strategy] += 1
        executor = StrategyExecutor(strategy)
        
        done = False
        turn = 0
        
        while not done and turn < env.T_max:
            action = executor.get_action(obs, turn)
            obs, reward, done, truncated, info = env.step(action)
            turn += 1
            
            if truncated:
                break
        
        terminal_type = info.get("last_transition_outcome", {}).get("terminal_type", "unknown")
        if terminal_type in outcomes:
            outcomes[terminal_type] += 1
        else:
            outcomes["other"] += 1
        
        turns_list.append(turn)
    
    resolution_rate = 100.0 * outcomes.get("resolved", 0) / n_episodes if n_episodes > 0 else 0
    escalation_rate = 100.0 * outcomes.get("escalated", 0) / n_episodes if n_episodes > 0 else 0
    
    return {
        "policy_type": "Strategy LinUCB (Fixed Strategy)",
        "outcomes": outcomes,
        "resolution_rate": resolution_rate,
        "escalation_rate": escalation_rate,
        "mean_turns": float(np.mean(turns_list)) if turns_list else 0,
        "strategy_distribution": {STRATEGIES[i]: int(strategy_dist[i]) for i in range(3)},
    }


def evaluate_per_turn_linucb(env: SupportEnv, policy: PerTurnLinUCBPolicy, n_episodes: int = 100) -> dict:
    """Evaluate Per-Turn LinUCB (independent decisions per turn)."""
    print(f"\nEvaluating Per-Turn LinUCB (independent per-turn decisions)...")
    
    outcomes = {"resolved": 0, "escalated": 0, "dropout": 0, "timeout": 0, "other": 0}
    turns_list = []
    
    for ep in range(n_episodes):
        if (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"  {ep + 1}/{n_episodes}")
        
        obs, _ = env.reset()
        done = False
        turn = 0
        
        while not done and turn < env.T_max:
            action = policy.predict_action(obs, turn)
            obs, reward, done, truncated, info = env.step(int(action))
            turn += 1
            
            if truncated:
                break
        
        terminal_type = info.get("last_transition_outcome", {}).get("terminal_type", "unknown")
        if terminal_type in outcomes:
            outcomes[terminal_type] += 1
        else:
            outcomes["other"] += 1
        
        turns_list.append(turn)
    
    resolution_rate = 100.0 * outcomes.get("resolved", 0) / n_episodes if n_episodes > 0 else 0
    escalation_rate = 100.0 * outcomes.get("escalated", 0) / n_episodes if n_episodes > 0 else 0
    
    return {
        "policy_type": "Per-Turn LinUCB (Independent Decisions)",
        "outcomes": outcomes,
        "resolution_rate": resolution_rate,
        "escalation_rate": escalation_rate,
        "mean_turns": float(np.mean(turns_list)) if turns_list else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare PPO vs Strategy LinUCB vs Per-Turn LinUCB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=150, help="Episodes per policy")
    parser.add_argument("--ppo-model", type=str, default="RL Out/models/best_model")
    parser.add_argument(
        "--strategy-linucb-model",
        type=str,
        default="Simulation_4/artifacts/strategy_linucb_models/policy.pkl",
    )
    parser.add_argument(
        "--per-turn-linucb-model",
        type=str,
        default="Simulation_4/artifacts/linucb_models_scaled/models",
    )
    parser.add_argument("--artifacts-root", type=str, default="Simulation_4/artifacts")
    parser.add_argument("--output", type=str, default="Simulation_4/artifacts/comparison_results.json")
    
    args = parser.parse_args()
    
    print("=" * 75)
    print("COMPARISON: PPO vs STRATEGY LinUCB vs PER-TURN LinUCB")
    print("=" * 75)
    
    # Initialize environment
    print(f"\n① Initializing environment...")
    env = build_bandit_env(
        artifacts_root=args.artifacts_root,
        use_nlp=True,
        use_nlg=False,
        use_masking=True,
    )
    print(f"✓ Environment ready")
    
    results = {}
    
    # Evaluate PPO
    print(f"\n② Evaluating PPO...")
    try:
        ppo_model = PPO.load(args.ppo_model)
        results["ppo"] = evaluate_ppo(env, ppo_model, args.episodes)
        print(f"✓ PPO evaluation complete")
    except Exception as e:
        print(f"✗ Failed to load PPO: {e}")
    
    # Evaluate Strategy LinUCB
    print(f"\n③ Evaluating Strategy LinUCB...")
    try:
        strategy_policy = StrategyLinUCB.load(args.strategy_linucb_model)
        results["strategy_linucb"] = evaluate_strategy_linucb(env, strategy_policy, args.episodes)
        print(f"✓ Strategy LinUCB evaluation complete")
    except Exception as e:
        print(f"✗ Failed to load Strategy LinUCB: {e}")
    
    # Evaluate Per-Turn LinUCB
    print(f"\n④ Evaluating Per-Turn LinUCB...")
    try:
        per_turn_policy = PerTurnLinUCBPolicy.load(args.per_turn_linucb_model)
        results["per_turn_linucb"] = evaluate_per_turn_linucb(env, per_turn_policy, args.episodes)
        print(f"✓ Per-Turn LinUCB evaluation complete")
    except Exception as e:
        print(f"✗ Failed to load Per-Turn LinUCB: {e}")
    
    # Print comparison
    print(f"\n" + "=" * 75)
    print("DETAILED COMPARISON")
    print("=" * 75)
    
    print(f"\n{'Policy':<35} {'Resolution':>15} {'Escalation':>15} {'Avg Turns':>15}")
    print("-" * 80)
    
    for policy_name in ["ppo", "strategy_linucb", "per_turn_linucb"]:
        if policy_name in results:
            res = results[policy_name]
            print(
                f"{res['policy_type']:<35} "
                f"{res['resolution_rate']:>14.1f}% "
                f"{res['escalation_rate']:>14.1f}% "
                f"{res['mean_turns']:>14.1f}"
            )
    
    # Detailed outcome analysis
    print(f"\n{'OUTCOME DISTRIBUTION':<40} {'Count':>10} {'Pct':>10}")
    print("=" * 60)
    
    for policy_name in ["ppo", "strategy_linucb", "per_turn_linucb"]:
        if policy_name in results:
            res = results[policy_name]
            print(f"\n{res['policy_type']}:")
            for outcome, count in res['outcomes'].items():
                pct = 100.0 * count / args.episodes
                print(f"  {outcome:20s}: {count:6d} ({pct:5.1f}%)")
    
    # Key findings
    print(f"\n" + "=" * 75)
    print("KEY FINDINGS")
    print("=" * 75)
    
    if "ppo" in results and "strategy_linucb" in results:
        ppo_res = results["ppo"]["resolution_rate"]
        strategy_res = results["strategy_linucb"]["resolution_rate"]
        gap = ppo_res - strategy_res
        
        print(f"\n1. Long-term vs Fixed Strategy:")
        print(f"   PPO (adaptive):     {ppo_res:.1f}%")
        print(f"   Strategy CB:        {strategy_res:.1f}%")
        print(f"   Gap:                {gap:.1f}%")
        
        if gap > 10:
            print(f"   ✓ Adaptive policy significantly outperforms fixed strategy")
        else:
            print(f"   ~ Performance gap is small (fixed strategy nearly optimal)")
    
    if "strategy_linucb" in results and "per_turn_linucb" in results:
        strategy_res = results["strategy_linucb"]["resolution_rate"]
        per_turn_res = results["per_turn_linucb"]["resolution_rate"]
        
        print(f"\n2. Fixed Strategy vs Per-Turn Independence:")
        print(f"   Strategy CB:        {strategy_res:.1f}%")
        print(f"   Per-Turn CB:        {per_turn_res:.1f}%")
        
        if strategy_res > per_turn_res + 5:
            print(f"   ✓ Fixed strategy better than per-turn (validates CB reframing)")
        else:
            print(f"   ~ Similar performance")
    
    # Save results
    print(f"\n⑤ Saving results...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Comparison results saved to {output_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
