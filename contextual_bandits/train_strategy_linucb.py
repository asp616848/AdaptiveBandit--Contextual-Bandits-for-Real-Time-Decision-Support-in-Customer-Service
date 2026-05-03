#!/usr/bin/env python3
"""
Train Strategy-based LinUCB on SupportEnv.

Select ONE strategy for entire conversation, then get final outcome.
True single-step CB problem.
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

from Simulation_4.contextual_bandits.env_factory import build_bandit_env
from Simulation_4.contextual_bandits.strategy_linucb import StrategyLinUCB, StrategyExecutor, STRATEGIES, STRATEGY_DESCRIPTIONS


def collect_strategy_episodes(
    env,
    n_episodes: int = 100,
) -> list[dict]:
    """
    Collect episodes where each episode uses ONE random strategy throughout.
    
    Args:
        env: SupportEnv
        n_episodes: Number of episodes
    
    Returns:
        List of episodes with context, strategy, outcome
    """
    episodes = []
    
    print(f"\nCollecting {n_episodes} strategy episodes (random strategy selection)...")
    
    for ep in range(n_episodes):
        if (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"  Episode {ep + 1}/{n_episodes}")
        
        # Random strategy selection
        strategy = np.random.randint(0, 3)
        executor = StrategyExecutor(strategy)
        
        obs, _ = env.reset()
        initial_obs = obs.copy()  # Use initial observation as context
        
        done = False
        turn = 0
        
        # Run entire episode with single strategy
        while not done and turn < env.T_max:
            action = executor.get_action(obs, turn)
            obs, reward, done, truncated, info = env.step(action)
            turn += 1
            
            if truncated:
                break
        
        # Determine reward (1 if resolved, 0 otherwise)
        terminal_type = info.get("last_transition_outcome", {}).get("terminal_type", "unknown")
        episode_reward = 1.0 if terminal_type == "success" else 0.0
        
        episodes.append({
            "episode": ep,
            "strategy": strategy,
            "strategy_name": STRATEGIES[strategy],
            "initial_obs": initial_obs.copy(),
            "outcome": terminal_type,
            "reward": episode_reward,
            "turns": turn,
        })
    
    print(f"✓ Collected {n_episodes} strategy episodes")
    return episodes


def train_strategy_linucb(
    episodes: list[dict],
    n_strategies: int = 3,
    d: int = 9,
    alpha: float = 1.0,
) -> StrategyLinUCB:
    """
    Train Strategy LinUCB from episodes.
    
    Args:
        episodes: Collected episodes
        n_strategies: Number of strategies
        d: Feature dimension
        alpha: Exploration parameter
    
    Returns:
        Trained StrategyLinUCB
    """
    print(f"\nTraining Strategy LinUCB...")
    
    policy = StrategyLinUCB(n_strategies, d, alpha)
    
    for episode in episodes:
        context = episode["initial_obs"]
        strategy = episode["strategy"]
        reward = episode["reward"]
        
        policy.update(context, strategy, reward)
    
    print(f"✓ Trained Strategy LinUCB")
    print(f"  Episodes: {len(episodes)}")
    
    stats = policy.get_stats()
    for s in range(n_strategies):
        count = int(stats["strategy_counts"][s])
        mean_reward = stats["mean_rewards_per_strategy"][s]
        print(f"  Strategy {s} ({STRATEGIES[s]:20s}): {count:3d} episodes, {mean_reward:.2f} mean reward")
    
    return policy


def evaluate_strategy_linucb(
    env,
    policy: StrategyLinUCB,
    n_episodes: int = 100,
) -> dict:
    """
    Evaluate Strategy LinUCB policy.
    
    Args:
        env: SupportEnv
        policy: Trained StrategyLinUCB
        n_episodes: Episodes to evaluate
    
    Returns:
        Evaluation results
    """
    print(f"\nEvaluating Strategy LinUCB on {n_episodes} episodes...")
    
    outcomes = {"resolved": 0, "escalated": 0, "dropout": 0, "timeout": 0, "other": 0}
    strategy_outcomes = {s: {"resolved": 0, "total": 0} for s in range(3)}
    
    for ep in range(n_episodes):
        if (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"  Episode {ep + 1}/{n_episodes}")
        
        obs, _ = env.reset()
        initial_obs = obs.copy()
        
        # Select strategy using learned policy (exploitation)
        strategy = policy.predict(initial_obs)
        executor = StrategyExecutor(strategy)
        
        done = False
        turn = 0
        
        while not done and turn < env.T_max:
            action = executor.get_action(obs, turn)
            obs, reward, done, truncated, info = env.step(action)
            turn += 1
            
            if truncated:
                break
        
        # Record outcome
        terminal_type = info.get("last_transition_outcome", {}).get("terminal_type", "unknown")
        if terminal_type in outcomes:
            outcomes[terminal_type] += 1
        else:
            outcomes["other"] += 1
        
        strategy_outcomes[strategy]["total"] += 1
        if terminal_type == "success":
            strategy_outcomes[strategy]["resolved"] += 1
    
    # Calculate metrics
    resolution_rate = 100.0 * outcomes.get("resolved", 0) / n_episodes if n_episodes > 0 else 0
    escalation_rate = 100.0 * outcomes.get("escalated", 0) / n_episodes if n_episodes > 0 else 0
    
    results = {
        "n_episodes": n_episodes,
        "outcomes": outcomes,
        "resolution_rate": resolution_rate,
        "escalation_rate": escalation_rate,
        "strategy_performance": {},
    }
    
    for s in range(3):
        total = strategy_outcomes[s]["total"]
        resolved = strategy_outcomes[s]["resolved"]
        res_rate = 100.0 * resolved / total if total > 0 else 0
        results["strategy_performance"][STRATEGIES[s]] = {
            "episodes": total,
            "resolved": resolved,
            "resolution_rate": res_rate,
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train Strategy-based LinUCB on SupportEnv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--collect-episodes",
        type=int,
        default=500,
        help="Episodes to collect for training",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=200,
        help="Episodes for evaluation",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Exploration parameter",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Simulation_4/artifacts/strategy_linucb_models",
        help="Output directory",
    )
    parser.add_argument(
        "--artifacts-root",
        type=str,
        default="Simulation_4/artifacts",
        help="Artifacts root for environment",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("STRATEGY-BASED LinUCB TRAINING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Collection episodes: {args.collect_episodes}")
    print(f"  Eval episodes: {args.eval_episodes}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Output: {output_dir}")
    print(f"\nStrategies:")
    for s in range(3):
        print(f"  {s}: {STRATEGIES[s]:20s} - {STRATEGY_DESCRIPTIONS[s]}")
    
    # Initialize environment
    print(f"\n① Initializing environment...")
    env = build_bandit_env(
        artifacts_root=args.artifacts_root,
        use_nlp=True,
        use_nlg=False,
        use_masking=True,
    )
    print(f"✓ Environment ready")
    
    # Collect episodes
    print(f"\n② Collecting episodes...")
    start_time = time.time()
    episodes = collect_strategy_episodes(env, args.collect_episodes)
    collection_time = time.time() - start_time
    print(f"✓ Collection completed in {collection_time:.1f}s")
    
    # Train
    print(f"\n③ Training Strategy LinUCB...")
    start_time = time.time()
    policy = train_strategy_linucb(episodes, n_strategies=3, d=9, alpha=args.alpha)
    training_time = time.time() - start_time
    print(f"✓ Training completed in {training_time:.1f}s")
    
    # Evaluate
    print(f"\n④ Evaluating policy...")
    start_time = time.time()
    results = evaluate_strategy_linucb(env, policy, args.eval_episodes)
    eval_time = time.time() - start_time
    print(f"✓ Evaluation completed in {eval_time:.1f}s")
    
    # Save
    print(f"\n⑤ Saving results...")
    policy.save(output_dir / "policy.pkl")
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Models saved to {output_dir}")
    
    # Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTiming:")
    print(f"  Collection: {collection_time:.1f}s")
    print(f"  Training:   {training_time:.1f}s")
    print(f"  Evaluation: {eval_time:.1f}s")
    print(f"\nEvaluation Results:")
    print(f"  Resolution Rate: {results['resolution_rate']:.1f}%")
    print(f"  Escalation Rate: {results['escalation_rate']:.1f}%")
    print(f"\nOutcome Distribution:")
    for outcome, count in results['outcomes'].items():
        pct = 100.0 * count / args.eval_episodes
        print(f"  {outcome:15s}: {count:3d} ({pct:5.1f}%)")
    print(f"\nStrategy Performance:")
    for strategy, perf in results['strategy_performance'].items():
        print(f"  {strategy:20s}: {perf['resolution_rate']:5.1f}% resolution ({perf['episodes']} episodes)")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
