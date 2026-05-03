#!/usr/bin/env python3
"""
Diagnostic: Debug the training vs eval gap for Phase 13 PPO.

Analyzes:
- Action distribution (is it stuck in AskInfo?)
- Turn patterns
- Episode trajectories
- Why eval (48.8%) < training (69.1%)
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
from collections import Counter

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Simulation_4.env.support_env import SupportEnv
from Simulation_4.training.action_masking import ActionMaskedEnv
from Simulation_4.training.nlp_observation import NLPObservationWrapper
from Simulation_4.env.agent_response_generator import AgentResponseGenerator
from stable_baselines3 import PPO


ACTION_NAMES = {
    0: "AskInfo",
    1: "ProvideSolution",
    2: "AffectiveRepair",
    3: "Escalate",
    4: "Close",
}


def analyze_trajectories(
    env: SupportEnv,
    model: PPO,
    n_episodes: int = 50,
) -> dict:
    """
    Analyze detailed trajectories to understand model behavior.
    """
    print(f"\nAnalyzing {n_episodes} trajectories...")
    
    results = {
        "action_counts": Counter(),
        "action_sequences": [],
        "episode_lengths": [],
        "turn_by_action": {i: Counter() for i in range(5)},
        "outcomes": Counter(),
        "avg_reward_per_turn": {i: [] for i in range(20)},
    }
    
    for ep in range(n_episodes):
        if (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"  Episode {ep + 1}/{n_episodes}")
        
        obs, _ = env.reset()
        done = False
        actions_taken = []
        turn = 0
        ep_reward = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            
            turn_count = int(env.state.get("turn_count", 0))
            results["turn_by_action"][action][turn_count] += 1
            
            obs, reward, done, truncated, info = env.step(action)
            
            actions_taken.append(action)
            results["action_counts"][action] += 1
            results["avg_reward_per_turn"][turn].append(float(reward))
            
            ep_reward += float(reward)
            turn += 1
            
            if truncated:
                break
        
        outcome = info.get("last_transition_outcome", {}).get("terminal_type", "unknown")
        results["outcomes"][outcome] += 1
        results["episode_lengths"].append(turn)
        results["action_sequences"].append([ACTION_NAMES[a] for a in actions_taken])
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Debug training vs eval gap for Phase 13",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RL Out/models/best_model",
        help="Path to trained PPO model",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of episodes to analyze",
    )
    parser.add_argument(
        "--artifacts-root",
        type=str,
        default="Simulation_4/artifacts",
        help="Artifacts root",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PHASE 13 GAP ANALYSIS: Training (69%) vs Eval (48.8%)")
    print("=" * 70)
    
    # Load environment with NLP wrapper (same as training)
    print(f"\n① Initializing environment with NLP wrapper...")
    env = SupportEnv(artifacts_root=args.artifacts_root, nlg_enabled=True)
    
    agent_gen = AgentResponseGenerator(model="llama3")
    env = NLPObservationWrapper(env, intent_model="phi3", agent_response_generator=agent_gen)
    env = ActionMaskedEnv(env)
    print(f"✓ Environment ready")
    
    # Load model
    print(f"\n② Loading PPO model from {args.model}...")
    model = PPO.load(args.model)
    print(f"✓ Model loaded")
    
    # Analyze trajectories
    print(f"\n③ Analyzing trajectories...")
    results = analyze_trajectories(env, model, n_episodes=args.episodes)
    
    # Print analysis
    print(f"\n" + "=" * 70)
    print("ACTION DISTRIBUTION")
    print("=" * 70)
    
    total_actions = sum(results["action_counts"].values())
    for action_id in range(5):
        count = results["action_counts"][action_id]
        pct = 100.0 * count / total_actions if total_actions > 0 else 0
        print(f"  {ACTION_NAMES[action_id]:20s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\n" + "=" * 70)
    print("TURN DISTRIBUTION BY ACTION")
    print("=" * 70)
    print("\nWhich turns use which actions? (High AskInfo at later turns = looping)")
    
    for action_id in range(5):
        turn_counts = results["turn_by_action"][action_id]
        if turn_counts:
            print(f"\n{ACTION_NAMES[action_id]:20s}:")
            for turn in sorted(turn_counts.keys())[:10]:
                count = turn_counts[turn]
                print(f"  Turn {turn}: {count:3d} times")
    
    print(f"\n" + "=" * 70)
    print("EPISODE STATISTICS")
    print("=" * 70)
    
    print(f"  Mean episode length:   {np.mean(results['episode_lengths']):.1f} turns")
    print(f"  Std episode length:    {np.std(results['episode_lengths']):.1f}")
    print(f"  Min/Max:               {np.min(results['episode_lengths'])}/{np.max(results['episode_lengths'])} turns")
    
    print(f"\n" + "=" * 70)
    print("OUTCOMES")
    print("=" * 70)
    
    total_ep = args.episodes
    for outcome, count in results["outcomes"].items():
        pct = 100.0 * count / total_ep
        print(f"  {outcome:20s}: {count:3d} ({pct:5.1f}%)")
    
    # Identify problematic patterns
    print(f"\n" + "=" * 70)
    print("DIAGNOSTIC FINDINGS")
    print("=" * 70)
    
    askinfo_pct = 100.0 * results["action_counts"][0] / total_actions if total_actions > 0 else 0
    
    if askinfo_pct > 60:
        print(f"\n⚠️  PROBLEM: {askinfo_pct:.1f}% of actions are AskInfo")
        print(f"    → Model is stuck in an AskInfo loop!")
        print(f"    → Why? Likely repeating the same questions")
        print(f"    → Solution: Need reward penalty for repeated AskInfo")
    
    elif askinfo_pct < 20:
        print(f"\n✓ Good: {askinfo_pct:.1f}% AskInfo (not looping)")
    
    # Check escalation
    esc_pct = 100.0 * results["action_counts"][3] / total_actions if total_actions > 0 else 0
    if esc_pct == 0:
        print(f"\n⚠️  CONSTRAINT: 0% Escalation during eval")
        print(f"    → Action masking is blocking it")
        print(f"    → Escalation (action 3) only allowed from turn 3+")
    
    # Compare to expected
    print(f"\n" + "=" * 70)
    print("HYPOTHESIS FOR 20% GAP")
    print("=" * 70)
    print(f"""
Training (69%): Used structured curriculum, reward shaping tuned
Eval (48.8%):   Using real LUMO scenarios with NLP overhead

Likely causes:
  1. NLP observations more noisy than training (different LLM calls)
  2. Action masking constraint (forced AskInfo) reduces effectiveness
  3. Evaluation scenarios harder than training distribution
  4. Model overfitting to training reward structure

To fix:
  A. Retrain WITH NLP wrapper and masking from start
  B. Increase turns before escalation allowed
  C. Add diversity penalty to prevent AskInfo loops
""")


if __name__ == "__main__":
    main()
