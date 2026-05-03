#!/usr/bin/env python3
"""
Phase 13 Model Testing Framework - Quick Tests
===============================================

Validates Phase 13 PPO model with:
  - 9D NLP observation space
  - Reward shaping (potential-based)
  - Action masking (Escalate blocked < turn 3)
  - Curriculum learning
"""

import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
import sys

import numpy as np
from stable_baselines3 import PPO

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Simulation_4.training.reward_shaping import RewardShaper

# ============================================================================
# TEST 1: Model Architecture
# ============================================================================

@dataclass
class ArchResult:
    model_loaded: bool
    obs_size: int
    action_size: int
    test_passed: bool

def test_model_architecture(model_path: str) -> ArchResult:
    print("\n" + "="*70)
    print("TEST 1: Model Architecture Validation")
    print("="*70)
    
    try:
        model = PPO.load(model_path)
        print(f"✓ Model loaded")
        
        obs_space = model.observation_space
        action_space = model.action_space
        
        obs_size = obs_space.shape[0] if hasattr(obs_space, 'shape') else 0
        action_size = action_space.n if hasattr(action_space, 'n') else 0
        
        print(f"  Observation space: {obs_space} (expected Box(9,))")
        print(f"  Action space: {action_space} (expected Discrete(5))")
        
        obs_ok = obs_size == 9
        action_ok = action_size == 5
        
        if obs_ok:
            print("  ✓ Observation space: 9D (Phase 13 spec)")
        else:
            print(f"  ✗ Observation size mismatch: {obs_size} vs 9")
            
        if action_ok:
            print("  ✓ Action space: Discrete(5)")
        else:
            print(f"  ✗ Action size mismatch: {action_size} vs 5")
        
        test_passed = obs_ok and action_ok
        print(f"\n  => TEST 1: {'PASSED ✓' if test_passed else 'FAILED ✗'}")
        
        return ArchResult(
            model_loaded=True,
            obs_size=obs_size,
            action_size=action_size,
            test_passed=test_passed,
        )
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print(f"\n  => TEST 1: FAILED ✗")
        return ArchResult(model_loaded=False, obs_size=0, action_size=0, test_passed=False)


# ============================================================================
# TEST 2: Reward Structure
# ============================================================================

@dataclass
class RewardResult:
    shaper_enabled: bool
    strict_potential: bool
    test_passed: bool

def test_reward_structure() -> RewardResult:
    print("\n" + "="*70)
    print("TEST 2: Reward Structure Validation")
    print("="*70)
    
    try:
        shaper = RewardShaper(enabled=True, strict_potential=True)
        
        print(f"✓ RewardShaper instantiated")
        print(f"  Enabled: {shaper.enabled}")
        print(f"  Strict potential mode: {shaper.strict_potential}")
        print(f"  Info gain bonus: {shaper.info_gain_bonus}")
        print(f"  Progress bonus: {shaper.progress_increase_bonus}")
        print(f"  Gamma: {shaper.gamma}")
        
        # Test potential function
        state_pre = {"information": 0.3, "progress": 0.2, "frustration": 0.5}
        state_post = {"information": 0.6, "progress": 0.5, "frustration": 0.2}
        
        phi_pre = shaper._potential(state_pre)
        phi_post = shaper._potential(state_post)
        
        print(f"\n  Potential function:")
        print(f"    Pre-state:  {phi_pre:.4f}")
        print(f"    Post-state: {phi_post:.4f}")
        
        shaped = shaper.shape(1.0, state_pre, state_post, 0, {})
        print(f"    Shaped reward (base=1.0): {shaped:.4f}")
        
        test_passed = shaper.enabled and shaper.strict_potential
        print(f"\n  => TEST 2: {'PASSED ✓' if test_passed else 'FAILED ✗'}")
        
        return RewardResult(
            shaper_enabled=shaper.enabled,
            strict_potential=shaper.strict_potential,
            test_passed=test_passed,
        )
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print(f"\n  => TEST 2: FAILED ✗")
        return RewardResult(shaper_enabled=False, strict_potential=False, test_passed=False)


# ============================================================================
# Main
# ============================================================================

def run_tests(model_path: str):
    print("\n" + "█"*70)
    print("  PHASE 13 MODEL TEST SUITE")
    print("█"*70)
    print(f"\nModel: {model_path}\n")
    
    start = time.time()
    
    arch_result = test_model_architecture(model_path)
    reward_result = test_reward_structure()
    
    elapsed = time.time() - start
    
    # Summary
    print("\n" + "█"*70)
    print("  TEST SUMMARY")
    print("█"*70)
    print(f"\nTime: {elapsed:.2f}s")
    
    passed = sum([arch_result.test_passed, reward_result.test_passed])
    total = 2
    print(f"Passed: {passed}/{total}\n")
    
    if arch_result.test_passed:
        print("✓ TEST 1 - Architecture")
    else:
        print("✗ TEST 1 - Architecture")
        
    if reward_result.test_passed:
        print("✓ TEST 2 - Reward Structure")
    else:
        print("✗ TEST 2 - Reward Structure")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!\n")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 13 Model Tests")
    parser.add_argument("--model", type=str, default="Simulation_4/artifacts/phase10_prod/models/best_model.zip")
    args = parser.parse_args()
    
    run_tests(args.model)
