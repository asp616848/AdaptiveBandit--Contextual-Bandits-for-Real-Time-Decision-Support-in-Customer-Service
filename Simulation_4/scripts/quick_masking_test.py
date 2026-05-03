#!/usr/bin/env python3
"""Quick test to verify action masking is working"""

import sys
from pathlib import Path

# Setup paths
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from Simulation_4.env.support_env import SupportEnv
import numpy as np

def test_action_masking():
    """Test that action masking blocks escalation before turn 3"""
    print("=" * 60)
    print("QUICK MASKING TEST")
    print("=" * 60)
    
    # Create environment
    artifacts_root = repo_root / "Simulation_4" / "artifacts"
    env = SupportEnv(str(artifacts_root), nlg_enabled=False)
    
    print(f"\n✓ Environment created successfully")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Observation space: {env.observation_space}")
    
    # Reset and test masking at different turns
    obs, info = env.reset()
    
    all_passed = True
    for step_num in range(5):
        print(f"\n--- Turn {step_num} ---")
        turn_count = int(env.state.get("turn_count", 0))
        print(f"State turn_count: {turn_count}")
        
        masks = env.action_masks()
        print(f"Action masks: {masks}")
        print(f"  Action 0 (AskInfo): {'✓ ALLOWED' if masks[0] else '✗ BLOCKED'}")
        print(f"  Action 1 (ProvideSolution): {'✓ ALLOWED' if masks[1] else '✗ BLOCKED'}")
        print(f"  Action 2 (AffectiveRepair): {'✓ ALLOWED' if masks[2] else '✗ BLOCKED'}")
        print(f"  Action 3 (Escalate): {'✓ ALLOWED' if masks[3] else '✗ BLOCKED'}")
        print(f"  Action 4 (Close): {'✓ ALLOWED' if masks[4] else '✗ BLOCKED'}")
        
        # Test escalation at different turns
        if hasattr(env, 'turn_count') and env.turn_count < 3:
            if masks[3]:  # Escalate should be blocked
                print(f"  ✗ FAIL: Escalate should be blocked at turn {turn_count}")
                all_passed = False
            else:
                print(f"  ✓ PASS: Escalate correctly blocked at turn {turn_count}")
        
        # Take a valid action to advance
        valid_action = 0  # AskInfo
        obs, reward, done, truncated, info = env.step(valid_action)
        
        if done:
            print("Episode ended, resetting...")
            obs, info = env.reset()
            break
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL MASKING TESTS PASSED")
    else:
        print("✗ SOME MASKING TESTS FAILED")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    test_action_masking()
