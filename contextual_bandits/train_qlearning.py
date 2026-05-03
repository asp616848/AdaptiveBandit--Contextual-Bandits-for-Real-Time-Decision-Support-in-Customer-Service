"""Train Q-Learning policy on SupportEnv."""

import argparse
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
from gymnasium import make

from Simulation_4.contextual_bandits.qlearning import QLearning
from Simulation_4.training.action_masking import ActionMaskedEnv
from Simulation_4.training.text_observation import NLPObservationWrapper


def train_qlearning(
    env_id: str = "SupportEnv",
    n_episodes: int = 1000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 0.2,
    epsilon_end: float = 0.05,
    state_bins: int = 32,
    output_dir: str | Path = "Simulation_4/artifacts/qlearning_models",
) -> QLearning:
    """Train Q-Learning agent."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    env = make(env_id)
    env = NLPObservationWrapper(env)
    env = ActionMaskedEnv(env)
    
    # Initialize Q-Learning
    model = QLearning(
        n_actions=env.action_space.n,
        n_state_bins=state_bins,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon_start,
    )
    
    # Training
    episode_rewards = []
    episode_lengths = []
    start_time = time.time()
    
    for ep in range(n_episodes):
        # Decay epsilon
        progress = ep / n_episodes
        model.epsilon = epsilon_start - (epsilon_start - epsilon_end) * progress
        
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        
        while not done:
            # Get action mask
            mask = info.get("action_mask", np.ones(env.action_space.n, dtype=bool))
            
            # Select action
            action = model.select_action(obs, mask)
            
            # Step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update Q-Learning
            model.update(obs, action, reward, next_obs, done)
            
            episode_reward += reward
            steps += 1
            obs = next_obs
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        if (ep + 1) % 100 == 0:
            elapsed = time.time() - start_time
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            print(
                f"Episode {ep + 1:4d}/{n_episodes} | "
                f"AvgReward: {avg_reward:7.3f} | "
                f"AvgLength: {avg_length:6.2f} | "
                f"TimeElapsed: {elapsed:7.1f}s"
            )
    
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s")
    
    # Save model
    model.save(output_dir / "model.pkl")
    
    # Save stats
    stats = {
        "algorithm": "Q-Learning",
        "n_episodes": n_episodes,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "state_bins": state_bins,
        "training_time": elapsed,
        "final_reward": float(episode_rewards[-1]),
        "avg_reward_last100": float(np.mean(episode_rewards[-100:])),
        "avg_length_last100": float(np.mean(episode_lengths[-100:])),
        "model_stats": model.get_stats(),
    }
    
    with open(output_dir / "training_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Model saved to: {output_dir / 'model.pkl'}")
    print(f"Stats saved to: {output_dir / 'training_stats.json'}")
    
    return model


def evaluate_qlearning(
    model: QLearning,
    env_id: str = "SupportEnv",
    n_episodes: int = 200,
) -> dict:
    """Evaluate Q-Learning policy."""
    
    env = make(env_id)
    env = NLPObservationWrapper(env)
    env = ActionMaskedEnv(env)
    
    results = {
        "resolutions": [],
        "escalations": [],
        "dropouts": [],
        "action_counts": Counter(),
        "episode_lengths": [],
    }
    
    start_time = time.time()
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        
        while not done:
            # Get action mask
            mask = info.get("action_mask", np.ones(env.action_space.n, dtype=bool))
            
            # Greedy action selection
            action = model.predict(obs, mask)
            results["action_counts"][action] += 1
            
            # Step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            obs = next_obs
        
        # Extract outcomes from wrapped env
        state = env.unwrapped.state.get()
        results["resolutions"].append(float(state.get("resolved", 0)))
        results["escalations"].append(float(state.get("escalated", 0)))
        results["dropouts"].append(float(state.get("customer_dropout", 0)))
        results["episode_lengths"].append(steps)
    
    elapsed = time.time() - start_time
    
    # Summary stats
    resolution_rate = np.mean(results["resolutions"]) * 100
    escalation_rate = np.mean(results["escalations"]) * 100
    dropout_rate = np.mean(results["dropouts"]) * 100
    
    print(f"\n{'='*60}")
    print(f"Q-Learning Evaluation ({n_episodes} episodes in {elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"Resolution Rate:  {resolution_rate:6.2f}%")
    print(f"Escalation Rate:  {escalation_rate:6.2f}%")
    print(f"Dropout Rate:     {dropout_rate:6.2f}%")
    print(f"Avg Episode Length: {np.mean(results['episode_lengths']):.1f}")
    print(f"\nAction Distribution:")
    action_names = ["AskInfo", "ProvideSolution", "AffectiveRepair", "Escalate", "Close"]
    for action_id, count in sorted(results["action_counts"].items()):
        pct = 100 * count / (n_episodes * 20)  # 20 = avg episode length
        print(f"  {action_names[action_id]:18s}: {count:5d} ({pct:5.1f}%)")
    
    return {
        "resolution_rate": resolution_rate,
        "escalation_rate": escalation_rate,
        "dropout_rate": dropout_rate,
        "avg_episode_length": float(np.mean(results["episode_lengths"])),
        "action_distribution": dict(results["action_counts"]),
        "evaluation_time": elapsed,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=1000, help="Training episodes")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=0.2, help="Initial epsilon")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--state-bins", type=int, default=32, help="State discretization")
    parser.add_argument("--eval-episodes", type=int, default=200, help="Evaluation episodes")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Simulation_4/artifacts/qlearning_models",
        help="Output directory",
    )
    args = parser.parse_args()
    
    # Train
    print(f"Training Q-Learning for {args.n_episodes} episodes...")
    model = train_qlearning(
        n_episodes=args.n_episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        state_bins=args.state_bins,
        output_dir=args.output_dir,
    )
    
    # Evaluate
    print(f"\nEvaluating Q-Learning on {args.eval_episodes} episodes...")
    results = evaluate_qlearning(model, n_episodes=args.eval_episodes)
    
    # Save results
    with open(Path(args.output_dir) / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {Path(args.output_dir) / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
