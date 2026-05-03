#!/usr/bin/env python3
"""
Diagnostic: Test each strategy independently to find the bottleneck.
"""

import sys
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Simulation_4.contextual_bandits.env_factory import build_bandit_env
from Simulation_4.contextual_bandits.strategy_linucb import StrategyExecutor, STRATEGIES
from Simulation_4.training.action_masking import ActionMaskedEnv
from Simulation_4.training.nlp_observation import NLPObservationWrapper


def test_strategy_independent(env, strategy_id, n_episodes=50):
    """Test a single strategy over many episodes."""
    
    outcomes = {"resolved": 0, "escalated": 0, "dropout": 0, "other": 0}
    actions_taken = []
    
    print(f"\nTesting Strategy {strategy_id}: {STRATEGIES[strategy_id]}")
    print("=" * 60)
    
    executor = StrategyExecutor(strategy_id)
    
    for ep in range(n_episodes):
        if (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"  Episode {ep + 1}/{n_episodes}")
        
        obs, _ = env.reset()
        done = False
        turn = 0
        ep_actions = []
        
        while not done and turn < env.T_max:
            # Get action from executor
            action = executor.get_action(obs, turn)
            ep_actions.append(action)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(int(action))
            turn += 1
            
            if truncated:
                break
        
        actions_taken.extend(ep_actions)
        
        # Record outcome
        terminal_type = info.get("last_transition_outcome", {}).get("terminal_type", "unknown")
        if terminal_type in outcomes:
            outcomes[terminal_type] += 1
        else:
            outcomes["other"] += 1
    
    # Calculate metrics
    resolution_rate = 100.0 * outcomes.get("resolved", 0) / n_episodes
    
    print(f"\nResults for {STRATEGIES[strategy_id]}:")
    print(f"  Resolution: {resolution_rate:.1f}%")
    print(f"  Escalation: {100.0 * outcomes.get('escalated', 0) / n_episodes:.1f}%")
    print(f"  Dropout: {100.0 * outcomes.get('dropout', 0) / n_episodes:.1f}%")
    print(f"  Other: {100.0 * outcomes.get('other', 0) / n_episodes:.1f}%")
    
    if actions_taken:
        unique_actions = set(actions_taken)
        print(f"  Actions used: {sorted(unique_actions)}")
        action_names = ["AskInfo", "ProvideSolution", "AffectiveRepair", "Escalate", "Close"]
        for action_id in sorted(unique_actions):
            count = actions_taken.count(action_id)
            pct = 100.0 * count / len(actions_taken)
            print(f"    {action_names[action_id]}: {pct:.1f}%")


def main():
    print("=" * 70)
    print("STRATEGY DIAGNOSTIC - Testing Each Strategy Independently")
    print("=" * 70)
    
    # Initialize environment
    print("\n① Initializing environment...")
    env = build_bandit_env(
        artifacts_root="Simulation_4/artifacts",
        use_nlp=True,
        use_nlg=False,
        use_masking=True,
    )
    
    # WRAP WITH OBSERVATION + MASKING (must match training!)
    print("  Wrapping with NLPObservationWrapper + ActionMaskedEnv...")
    env = NLPObservationWrapper(env)
    env = ActionMaskedEnv(env)
    
    print("✓ Environment ready with proper wrappers")
    
    # Test each strategy
    print("\n② Testing strategies...")

    test_strategy_independent(env, 0, 50)  # escalate_fast
    test_strategy_independent(env, 1, 50)  # solve_patiently
    test_strategy_independent(env, 2, 50)  # adaptive_repair
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
