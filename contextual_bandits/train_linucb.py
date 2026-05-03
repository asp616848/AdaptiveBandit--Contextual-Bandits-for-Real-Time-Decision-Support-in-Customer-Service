#!/usr/bin/env python3
"""
Train Per-Turn LinUCB on SupportEnv simulator.

Collects episodes from the simulator with random exploration,
then trains separate LinUCB models for each turn.
"""

import argparse
import sys
from pathlib import Path
from typing import Any
import json
import time
import numpy as np

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Simulation_4.env.support_env import SupportEnv
from Simulation_4.contextual_bandits.linucb import PerTurnLinUCBPolicy


def collect_episodes(
    env: SupportEnv,
    n_episodes: int = 100,
    policy: str = "random",
    epsilon: float = 0.1,
) -> list[dict[str, Any]]:
    """
    Collect episodes from environment for training.
    
    Args:
        env: SupportEnv instance
        n_episodes: Number of episodes to collect
        policy: "random" or "epsilon-greedy"
        epsilon: Exploration rate
    
    Returns:
        List of episodes with turns, observations, actions, rewards
    """
    episodes = []
    
    print(f"\nCollecting {n_episodes} episodes with {policy} policy...")
    
    for ep in range(n_episodes):
        if (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"  Episode {ep + 1}/{n_episodes}")
        
        obs, _ = env.reset()
        done = False
        turns = []
        
        while not done:
            turn_count = int(env.state.get("turn_count", 0))
            
            # Select action
            if policy == "random":
                action = env.action_space.sample()
            elif policy == "epsilon-greedy":
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    # Use random for warmup
                    action = env.action_space.sample()
            else:
                action = env.action_space.sample()
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
            # Record turn data
            turn_data = {
                "turn": turn_count,
                "obs": obs.copy(),
                "action": int(action),
                "reward": float(reward),
            }
            turns.append(turn_data)
            
            if truncated:
                break
        
        episodes.append({
            "episode": ep,
            "turns": turns,
            "terminal_type": info.get("last_transition_outcome", {}).get("terminal_type", "unknown"),
        })
    
    print(f"✓ Collected {n_episodes} episodes")
    return episodes


def train_linucb(
    episodes: list[dict[str, Any]],
    n_turns: int = 20,
    n_actions: int = 5,
    d: int = 9,
    alpha: float = 1.0,
) -> PerTurnLinUCBPolicy:
    """
    Train per-turn LinUCB models from episodes.
    
    Args:
        episodes: List of episodes with turns, observations, actions, rewards
        n_turns: Max number of turns
        n_actions: Number of actions
        d: Feature dimension
        alpha: Exploration parameter
    
    Returns:
        Trained PerTurnLinUCBPolicy
    """
    print(f"\nTraining per-turn LinUCB models ({n_turns} turns)...")
    
    policy = PerTurnLinUCBPolicy(n_turns, n_actions, d, alpha)
    
    total_updates = 0
    for ep, episode in enumerate(episodes):
        for turn_data in episode["turns"]:
            turn = turn_data["turn"]
            obs = turn_data["obs"]
            action = turn_data["action"]
            reward = turn_data["reward"]
            
            # Update per-turn model
            policy.update(obs, turn, action, reward)
            total_updates += 1
    
    print(f"✓ Trained {n_turns} per-turn models")
    print(f"  Total parameter updates: {total_updates:,}")
    
    return policy


def evaluate_linucb(
    env: SupportEnv,
    policy: PerTurnLinUCBPolicy,
    n_episodes: int = 50,
) -> dict[str, Any]:
    """
    Evaluate LinUCB policy on environment.
    
    Args:
        env: SupportEnv instance
        policy: PerTurnLinUCBPolicy
        n_episodes: Number of episodes to evaluate
    
    Returns:
        Evaluation results
    """
    print(f"\nEvaluating LinUCB on {n_episodes} episodes...")
    
    outcomes = {
        "resolved": 0,
        "escalated": 0,
        "dropout": 0,
        "timeout": 0,
        "other": 0,
    }
    
    rewards = []
    turns_list = []
    
    for ep in range(n_episodes):
        if (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"  Episode {ep + 1}/{n_episodes}")
        
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        n_turns = 0
        
        while not done:
            turn_count = int(env.state.get("turn_count", 0))
            
            # Select action using LinUCB (exploitation mode)
            action = policy.predict_action(obs, turn_count)
            
            obs, reward, done, truncated, info = env.step(int(action))
            ep_reward += float(reward)
            n_turns += 1
            
            if truncated:
                break
        
        # Record episode outcome
        terminal_type = info.get("last_transition_outcome", {}).get("terminal_type", "other")
        if terminal_type in outcomes:
            outcomes[terminal_type] += 1
        else:
            outcomes["other"] += 1
        
        rewards.append(ep_reward)
        turns_list.append(n_turns)
    
    # Calculate metrics
    total_episodes = n_episodes
    resolution_rate = 100.0 * outcomes.get("resolved", 0) / total_episodes if total_episodes > 0 else 0
    escalation_rate = 100.0 * outcomes.get("escalated", 0) / total_episodes if total_episodes > 0 else 0
    
    results = {
        "n_episodes": n_episodes,
        "outcomes": outcomes,
        "resolution_rate": resolution_rate,
        "escalation_rate": escalation_rate,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_turns": float(np.mean(turns_list)),
        "std_turns": float(np.std(turns_list)),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train Per-Turn LinUCB on SupportEnv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--collect-episodes",
        type=int,
        default=200,
        help="Number of episodes to collect for training",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Number of episodes for evaluation",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Exploration parameter for LinUCB",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Simulation_4/artifacts/linucb_models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--artifacts-root",
        type=str,
        default="Simulation_4/artifacts",
        help="Artifacts root for environment",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PER-TURN LinUCB TRAINING ON SupportEnv SIMULATOR")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Collection episodes: {args.collect_episodes}")
    print(f"  Eval episodes: {args.eval_episodes}")
    print(f"  Alpha (exploration): {args.alpha}")
    print(f"  Output directory: {output_dir}")
    
    # Initialize environment
    print(f"\n① Initializing environment...")
    env = SupportEnv(artifacts_root=args.artifacts_root, nlg_enabled=False)
    print(f"✓ Environment ready (action_space={env.action_space}, obs_space={env.observation_space})")
    
    # Collect episodes
    print(f"\n② Collecting episodes with random exploration...")
    start_time = time.time()
    episodes = collect_episodes(env, n_episodes=args.collect_episodes, policy="random")
    collection_time = time.time() - start_time
    print(f"✓ Collection completed in {collection_time:.1f}s")
    
    # Train LinUCB
    print(f"\n③ Training per-turn LinUCB...")
    start_time = time.time()
    policy = train_linucb(
        episodes,
        n_turns=20,
        n_actions=5,
        d=9,
        alpha=args.alpha,
    )
    training_time = time.time() - start_time
    print(f"✓ Training completed in {training_time:.1f}s")
    
    # Evaluate
    print(f"\n④ Evaluating policy on environment...")
    start_time = time.time()
    results = evaluate_linucb(env, policy, n_episodes=args.eval_episodes)
    eval_time = time.time() - start_time
    print(f"✓ Evaluation completed in {eval_time:.1f}s")
    
    # Save results
    print(f"\n⑤ Saving results...")
    
    # Save models
    policy.save(output_dir / "models")
    print(f"✓ Models saved to {output_dir}/models")
    
    # Save evaluation results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {results_file}")
    
    # Print summary
    print(f"\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"\nCollection: {args.collect_episodes} episodes in {collection_time:.1f}s")
    print(f"Training:   {training_time:.1f}s")
    print(f"Evaluation: {args.eval_episodes} episodes in {eval_time:.1f}s")
    print(f"\nEvaluation Results:")
    print(f"  Resolution Rate:  {results['resolution_rate']:6.1f}%")
    print(f"  Escalation Rate:  {results['escalation_rate']:6.1f}%")
    print(f"  Mean Reward:      {results['mean_reward']:6.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Turns:       {results['mean_turns']:6.1f} ± {results['std_turns']:.1f}")
    print(f"\nOutcome Distribution:")
    for outcome, count in results['outcomes'].items():
        pct = 100.0 * count / args.eval_episodes
        print(f"  {outcome:15s}: {count:3d} ({pct:5.1f}%)")
    print("=" * 70)
    
    # Comparison note
    print(f"\nComparison to PPO:")
    print(f"  PPO (full MDP)       ~69% resolution")
    print(f"  LinUCB (per-turn)    {results['resolution_rate']:.1f}% resolution")
    print(f"  Expected difference: LinUCB lower (ignores sequential structure)")


if __name__ == "__main__":
    main()
