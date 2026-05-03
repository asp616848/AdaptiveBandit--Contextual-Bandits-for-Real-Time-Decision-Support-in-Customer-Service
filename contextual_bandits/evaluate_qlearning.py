"""Compare Q-Learning vs PPO policies."""

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
from gymnasium import make
from stable_baselines3 import PPO

from Simulation_4.contextual_bandits.qlearning import QLearning
from Simulation_4.training.action_masking import ActionMaskedEnv
from Simulation_4.training.text_observation import NLPObservationWrapper


def evaluate_policy(
    model,
    model_type: str = "qlearning",
    env_id: str = "SupportEnv",
    n_episodes: int = 100,
) -> dict:
    """Evaluate policy side-by-side."""
    
    env = make(env_id)
    env = NLPObservationWrapper(env)
    env = ActionMaskedEnv(env)
    
    results = {
        "resolutions": [],
        "escalations": [],
        "dropouts": [],
        "action_counts": Counter(),
        "turn_by_action": {i: Counter() for i in range(5)},
        "episode_lengths": [],
    }
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done:
            # Get action mask
            mask = info.get("action_mask", np.ones(env.action_space.n, dtype=bool))
            
            if model_type == "qlearning":
                action = model.predict(obs, mask)
            elif model_type == "ppo":
                action, _ = model.predict(obs, deterministic=True)
                # Apply masking
                if not mask[action]:
                    valid_actions = np.where(mask)[0]
                    action = int(np.random.choice(valid_actions))
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Track by turn
            turn = env.unwrapped.state.get("turn_count", 0)
            results["turn_by_action"][turn][action] += 1
            results["action_counts"][action] += 1
            
            # Step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            steps += 1
            obs = next_obs
        
        # Extract outcomes
        state = env.unwrapped.state.get()
        results["resolutions"].append(float(state.get("resolved", 0)))
        results["escalations"].append(float(state.get("escalated", 0)))
        results["dropouts"].append(float(state.get("customer_dropout", 0)))
        results["episode_lengths"].append(steps)
    
    # Summary stats
    return {
        "model_type": model_type,
        "n_episodes": n_episodes,
        "resolution_rate": float(np.mean(results["resolutions"]) * 100),
        "escalation_rate": float(np.mean(results["escalations"]) * 100),
        "dropout_rate": float(np.mean(results["dropouts"]) * 100),
        "avg_episode_length": float(np.mean(results["episode_lengths"])),
        "action_distribution": {k: v for k, v in results["action_counts"].items()},
        "turn_by_action": {str(k): {str(a): c for a, c in v.items()} for k, v in results["turn_by_action"].items()},
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qlearning-model",
        type=str,
        default="Simulation_4/artifacts/qlearning_models/model.pkl",
        help="Path to Q-Learning model",
    )
    parser.add_argument(
        "--ppo-model",
        type=str,
        default="best_model/policy.pth",
        help="Path to PPO model",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Evaluation episodes per model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Simulation_4/artifacts/comparison_results",
        help="Output directory",
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("Loading models...")
    qlearning_path = Path(args.qlearning_model)
    ppo_path = Path(args.ppo_model)
    
    if not qlearning_path.exists():
        print(f"ERROR: Q-Learning model not found at {qlearning_path}")
        print(f"Try: python -m Simulation_4.contextual_bandits.train_qlearning")
        return
    
    ql_model = QLearning.load(qlearning_path)
    print(f"✓ Loaded Q-Learning model ({len(ql_model.Q)} state-action pairs)")
    
    if not ppo_path.exists():
        print(f"ERROR: PPO model not found at {ppo_path}")
        return
    
    ppo_model = PPO.load(ppo_path)
    print(f"✓ Loaded PPO model")
    
    # Evaluate
    print(f"\nEvaluating both models on {args.n_episodes} episodes each...")
    print("-" * 60)
    
    ql_results = evaluate_policy(
        ql_model,
        model_type="qlearning",
        n_episodes=args.n_episodes,
    )
    
    ppo_results = evaluate_policy(
        ppo_model,
        model_type="ppo",
        n_episodes=args.n_episodes,
    )
    
    # Display results
    print(f"\n{'Algorithm':<20} {'Resolution':<15} {'Escalation':<15} {'Dropout':<15}")
    print("-" * 60)
    print(
        f"{'Q-Learning':<20} {ql_results['resolution_rate']:>6.2f}%           "
        f"{ql_results['escalation_rate']:>6.2f}%           "
        f"{ql_results['dropout_rate']:>6.2f}%"
    )
    print(
        f"{'PPO':<20} {ppo_results['resolution_rate']:>6.2f}%           "
        f"{ppo_results['escalation_rate']:>6.2f}%           "
        f"{ppo_results['dropout_rate']:>6.2f}%"
    )
    print("-" * 60)
    
    gap = ppo_results['resolution_rate'] - ql_results['resolution_rate']
    print(f"\nPPO vs Q-Learning: +{gap:6.2f}% resolution advantage")
    
    # Save comparison
    comparison = {
        "qlearning": ql_results,
        "ppo": ppo_results,
        "gap_resolution": gap,
    }
    
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nComparison saved to: {output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
