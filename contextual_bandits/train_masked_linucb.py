#!/usr/bin/env python3
"""
Train Per-Turn LinUCB that RESPECTS ACTION MASKING (fair PPO comparison).

This is the corrected version - samples only VALID actions during training.
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


def collect_episodes_masked(
    env: SupportEnv,
    n_episodes: int = 100,
) -> list[dict[str, Any]]:
    """
    Collect episodes, respecting ACTION MASKING (FIXED VERSION).
    
    Args:
        env: SupportEnv instance
        n_episodes: Number of episodes
    
    Returns:
        List of episodes with turns, observations, valid actions, rewards
    """
    episodes = []
    
    print(f"\nCollecting {n_episodes} episodes with MASKED ACTION SAMPLING...")
    
    for ep in range(n_episodes):
        if (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"  Episode {ep + 1}/{n_episodes}")
        
        obs, _ = env.reset()
        done = False
        turns = []
        masked_actions_count = 0
        
        while not done:
            turn_count = int(env.state.get("turn_count", 0))
            
            # GET ACTION MASK from environment
            action_mask = env.action_masks()  # Boolean array [True, True, True, False, True] etc.
            valid_actions = np.where(action_mask)[0]  # Get valid action indices
            
            # Sample ONLY from valid actions
            action = np.random.choice(valid_actions)
            
            if not action_mask[action]:
                masked_actions_count += 1
            
            # Step environment
            obs, reward, done, truncated, info = env.step(int(action))
            
            # Record turn data
            turn_data = {
                "turn": turn_count,
                "obs": obs.copy(),
                "action": int(action),
                "reward": float(reward),
                "valid_actions": list(valid_actions),
            }
            turns.append(turn_data)
            
            if truncated:
                break
        
        episodes.append({
            "episode": ep,
            "turns": turns,
            "terminal_type": info.get("last_transition_outcome", {}).get("terminal_type", "unknown"),
            "masked_actions_attempted": masked_actions_count,
        })
    
    print(f"✓ Collected {n_episodes} episodes with MASKED action sampling")
    return episodes


def train_per_turn_linucb_masked(
    episodes: list[dict[str, Any]],
) -> PerTurnLinUCBPolicy:
    """
    Train Per-Turn LinUCB from masked-action episodes.
    
    Args:
        episodes: Collected episodes
    
    Returns:
        Trained PerTurnLinUCBPolicy
    """
    print(f"\nTraining Per-Turn LinUCB (masked actions)...")
    
    policy = PerTurnLinUCBPolicy(n_turns=20, d=9, alpha=1.0)
    
    total_updates = 0
    for episode in episodes:
        for turn_data in episode["turns"]:
            turn = turn_data["turn"]
            context = turn_data["obs"]
            action = turn_data["action"]
            reward = turn_data["reward"]
            
            policy.update(context, turn, action, reward)
            total_updates += 1
    
    print(f"✓ Trained Per-Turn LinUCB (masked)")
    print(f"  Total parameter updates: {total_updates}")
    
    return policy


def evaluate_per_turn_linucb_masked(
    env: SupportEnv,
    policy: PerTurnLinUCBPolicy,
    n_episodes: int = 100,
) -> dict:
    """
    Evaluate Per-Turn LinUCB (masked actions, exploitation only).
    
    Args:
        env: SupportEnv
        policy: Trained PerTurnLinUCBPolicy
        n_episodes: Episodes to evaluate
    
    Returns:
        Evaluation results
    """
    print(f"\nEvaluating Per-Turn LinUCB (masked) on {n_episodes} episodes...")
    
    outcomes = {"resolved": 0, "escalated": 0, "dropout": 0, "timeout": 0, "other": 0}
    turns_list = []
    invalid_action_attempts = 0
    
    for ep in range(n_episodes):
        if (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"  Episode {ep + 1}/{n_episodes}")
        
        obs, _ = env.reset()
        done = False
        turn = 0
        
        while not done and turn < env.T_max:
            # Get valid actions
            action_mask = env.action_masks()
            valid_actions = np.where(action_mask)[0]
            
            # Use policy to predict best action
            predicted_action = policy.predict_action(obs, turn)
            
            # If predicted action is masked, select best valid action instead
            if not action_mask[predicted_action]:
                # Find best valid action according to policy
                best_value = -np.inf
                best_action = valid_actions[0]
                
                for valid_action in valid_actions:
                    # Policy's internal value for this action (LinUCB arm)
                    arm = policy.models[turn].arms[valid_action]
                    try:
                        theta = np.linalg.solve(arm.A, arm.b)
                        value = obs @ theta
                        if value > best_value:
                            best_value = value
                            best_action = valid_action
                    except (np.linalg.LinAlgError, ValueError):
                        # Singular matrix or invalid - just use this action
                        best_action = valid_action
                        break
                
                predicted_action = best_action
                invalid_action_attempts += 1
            
            obs, reward, done, truncated, info = env.step(int(predicted_action))
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
    
    results = {
        "policy_type": "Per-Turn LinUCB (Masked Actions)",
        "outcomes": outcomes,
        "resolution_rate": resolution_rate,
        "escalation_rate": escalation_rate,
        "mean_turns": float(np.mean(turns_list)) if turns_list else 0,
        "invalid_action_attempts": invalid_action_attempts,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train Masked Per-Turn LinUCB (CORRECTED VERSION)",
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
        "--output-dir",
        type=str,
        default="Simulation_4/artifacts/masked_linucb_models",
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
    print("MASKED PER-TURN LinUCB TRAINING (CORRECTED)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Collection episodes: {args.collect_episodes}")
    print(f"  Eval episodes: {args.eval_episodes}")
    print(f"  Output: {output_dir}")
    print(f"\nKey Difference:")
    print(f"  ✓ Samples ONLY valid actions (respects action masking)")
    print(f"  ✓ Creates properly aligned training data")
    print(f"  ✓ Fair comparison with PPO")
    
    # Initialize environment
    print(f"\n① Initializing environment...")
    env = SupportEnv(artifacts_root=args.artifacts_root, nlg_enabled=False)
    print(f"✓ Environment ready")
    
    # Collect episodes (with masked action sampling)
    print(f"\n② Collecting episodes with action masking...")
    start_time = time.time()
    episodes = collect_episodes_masked(env, args.collect_episodes)
    collection_time = time.time() - start_time
    print(f"✓ Collection completed in {collection_time:.1f}s")
    
    # Train
    print(f"\n③ Training Per-Turn LinUCB (masked)...")
    start_time = time.time()
    policy = train_per_turn_linucb_masked(episodes)
    training_time = time.time() - start_time
    print(f"✓ Training completed in {training_time:.1f}s")
    
    # Evaluate
    print(f"\n④ Evaluating policy...")
    start_time = time.time()
    results = evaluate_per_turn_linucb_masked(env, policy, args.eval_episodes)
    eval_time = time.time() - start_time
    print(f"✓ Evaluation completed in {eval_time:.1f}s")
    
    # Save
    print(f"\n⑤ Saving results...")
    policy.save(output_dir / "models")
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Models saved to {output_dir}")
    
    # Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY - MASKED PER-TURN LinUCB")
    print("=" * 70)
    print(f"\nTiming:")
    print(f"  Collection: {collection_time:.1f}s")
    print(f"  Training:   {training_time:.1f}s")
    print(f"  Evaluation: {eval_time:.1f}s")
    print(f"\nEvaluation Results:")
    print(f"  Resolution Rate: {results['resolution_rate']:.1f}%")
    print(f"  Escalation Rate: {results['escalation_rate']:.1f}%")
    print(f"  Avg Turns: {results['mean_turns']:.1f}")
    print(f"\nOutcome Distribution:")
    for outcome, count in results['outcomes'].items():
        pct = 100.0 * count / args.eval_episodes
        print(f"  {outcome:15s}: {count:3d} ({pct:5.1f}%)")
    
    if results['invalid_action_attempts'] > 0:
        print(f"\n⚠ Invalid action attempts during eval: {results['invalid_action_attempts']}")
        print(f"  (Policy tried masked actions, fallback to valid actions)")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
