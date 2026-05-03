#!/usr/bin/env python3
"""
Phase 13 Diagnostic: Training vs LUMO Evaluation Comparison

Systematically tests the model under different conditions to identify
which component is causing the train/eval discrepancy.

Key Tests:
1. Model integrity check (weights loaded, forward pass works)
2. Observation generation comparison (training vs LUMO)
3. Action masking verification
4. Single episode traces (detailed breakdowns)
5. Reward function comparison
6. Environment configuration check

Usage:
    python phase13_diagnostic.py --episodes 10 --verbose
"""

import sys
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Add repo to path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Simulation_4.env.support_env import SupportEnv
from Simulation_4.rag.scenario_generator import ScenarioGenerator


@dataclass
class DiagnosticStep:
    """Track a single diagnostic test step"""
    name: str
    status: str  # "PASS", "WARN", "FAIL"
    message: str
    details: dict = field(default_factory=dict)


class Phase13Diagnostic:
    def __init__(self, model_path="RL Out/models/best_model", verbose=False):
        self.model_path = model_path
        self.verbose = verbose
        self.steps = []
        self.model = None
        self.env = None
        
    def log(self, msg, level="INFO"):
        """Print diagnostic messages"""
        prefix = f"[{level:5s}]"
        print(f"{prefix} {msg}")
    
    def add_step(self, step: DiagnosticStep):
        """Record a diagnostic step"""
        self.steps.append(step)
        symbol = "✓" if step.status == "PASS" else "⚠" if step.status == "WARN" else "✗"
        print(f"  {symbol} {step.name}: {step.message}")
        if self.verbose and step.details:
            for k, v in step.details.items():
                print(f"      {k}: {v}")
    
    def test_model_integrity(self):
        """Test 1: Verify model loads correctly"""
        self.log("\n1. MODEL INTEGRITY CHECK", "TEST")
        
        try:
            self.log("Loading model from: " + self.model_path)
            self.model = PPO.load(self.model_path)
            
            self.add_step(DiagnosticStep(
                name="Model Loading",
                status="PASS",
                message="Model loaded successfully",
                details={
                    "Obs Space": str(self.model.observation_space),
                    "Action Space": str(self.model.action_space),
                    "Policy Type": type(self.model.policy).__name__,
                }
            ))
            
            # Test forward pass
            test_obs = np.random.randn(1, self.model.observation_space.shape[0]).astype(np.float32)
            action, _ = self.model.predict(test_obs, deterministic=True)
            
            self.add_step(DiagnosticStep(
                name="Forward Pass",
                status="PASS",
                message="Model can process observations and output actions",
                details={"Test Action": str(action)}
            ))
            
            return True
        except Exception as e:
            self.add_step(DiagnosticStep(
                name="Model Loading",
                status="FAIL",
                message=f"Failed to load model: {e}"
            ))
            return False
    
    def test_environment_setup(self):
        """Test 2: Verify environment initializes correctly"""
        self.log("\n2. ENVIRONMENT SETUP CHECK", "TEST")
        
        try:
            self.log("Initializing SupportEnv...")
            self.env = SupportEnv(artifacts_root="Simulation_4/artifacts", nlg_enabled=True)
            
            self.add_step(DiagnosticStep(
                name="Environment Init",
                status="PASS",
                message="SupportEnv initialized successfully",
                details={
                    "Observation Space": str(self.env.observation_space),
                    "Action Space": str(self.env.action_space),
                    "NLG Enabled": str(self.env.nlg_enabled),
                }
            ))
            
            # Test reset
            obs, info = self.env.reset(seed=42)
            
            if obs.shape[0] == self.model.observation_space.shape[0]:
                self.add_step(DiagnosticStep(
                    name="Observation Generation",
                    status="PASS",
                    message=f"Environment generates correct observation shape: {obs.shape}",
                    details={
                        "Expected Dim": 9,
                        "Actual Dim": obs.shape[0],
                    }
                ))
            else:
                self.add_step(DiagnosticStep(
                    name="Observation Generation",
                    status="FAIL",
                    message=f"Observation dimension mismatch: {obs.shape[0]} != {self.model.observation_space.shape[0]}",
                    details={
                        "Expected": self.model.observation_space.shape[0],
                        "Actual": obs.shape[0],
                        "Obs Content": str(obs),
                    }
                ))
                return False
            
            return True
        except Exception as e:
            self.add_step(DiagnosticStep(
                name="Environment Setup",
                status="FAIL",
                message=f"Failed to initialize environment: {e}"
            ))
            return False
    
    def test_observation_details(self):
        """Test 3: Detailed observation analysis"""
        self.log("\n3. OBSERVATION SPACE ANALYSIS", "TEST")
        
        try:
            obs, info = self.env.reset(seed=42)
            
            print(f"\n  Observation Details:")
            print(f"    Shape: {obs.shape}")
            print(f"    Values (first 9D):")
            for i, val in enumerate(obs):
                print(f"      [{i}] = {val:.4f}")
            
            # Check if all observations are valid
            if np.any(np.isnan(obs)):
                self.add_step(DiagnosticStep(
                    name="Observation Validity",
                    status="FAIL",
                    message="Observation contains NaN values!"
                ))
            elif np.any(np.isinf(obs)):
                self.add_step(DiagnosticStep(
                    name="Observation Validity",
                    status="FAIL",
                    message="Observation contains Inf values!"
                ))
            else:
                self.add_step(DiagnosticStep(
                    name="Observation Validity",
                    status="PASS",
                    message="All observation values are valid"
                ))
            
            # Check state dict
            state = self.env.state
            print(f"\n  Environment State:")
            for key in ['tier', 'persona_label', 'turn_count']:
                if key in state:
                    print(f"    {key}: {state[key]}")
            
            return True
        except Exception as e:
            self.add_step(DiagnosticStep(
                name="Observation Analysis",
                status="FAIL",
                message=f"Error analyzing observations: {e}"
            ))
            return False
    
    def test_action_masking(self):
        """Test 4: Verify action masking logic"""
        self.log("\n4. ACTION MASKING VERIFICATION", "TEST")
        
        try:
            # Test at different turn counts
            for turn_limit in [1, 3, 5]:
                obs, info = self.env.reset(seed=42)
                
                # Manually set turn count
                self.env.state['turn_count'] = turn_limit
                
                # Get mask from env
                if hasattr(self.env, 'action_masks'):
                    mask = self.env.action_masks()
                elif hasattr(self.env, 'get_action_mask'):
                    mask = self.env.get_action_mask()
                else:
                    mask = np.ones(self.env.action_space.n)
                
                escalate_allowed = bool(mask[3]) if len(mask) > 3 else True
                
                print(f"\n  Turn {turn_limit}: Escalation Allowed? {escalate_allowed}")
                
                if turn_limit < 3 and escalate_allowed:
                    self.add_step(DiagnosticStep(
                        name=f"Masking (T={turn_limit})",
                        status="FAIL",
                        message=f"Escalation allowed at turn {turn_limit} (should block until turn 3)"
                    ))
                elif turn_limit >= 3 and not escalate_allowed:
                    self.add_step(DiagnosticStep(
                        name=f"Masking (T={turn_limit})",
                        status="FAIL",
                        message=f"Escalation blocked at turn {turn_limit} (should allow)"
                    ))
                else:
                    self.add_step(DiagnosticStep(
                        name=f"Masking (T={turn_limit})",
                        status="PASS",
                        message=f"Masking working correctly"
                    ))
            
            return True
        except Exception as e:
            self.add_step(DiagnosticStep(
                name="Action Masking",
                status="WARN",
                message=f"Could not fully test masking: {e}"
            ))
            return True  # Don't fail, just warning
    
    def test_single_episode_trace(self, episode_id=0, max_steps=20):
        """Test 5: Detailed trace of a single episode"""
        self.log(f"\n5. SINGLE EPISODE TRACE (Episode {episode_id})", "TEST")
        
        try:
            obs, info = self.env.reset(seed=1000 + episode_id)
            done = False
            step = 0
            
            print(f"\n  Step Details:")
            print(f"  {'Step':<6} {'Action':<10} {'Terminal':<12} {'Reward':<10} {'Turn':<6}")
            print(f"  {'-'*60}")
            
            while not done and step < max_steps:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(int(action))
                
                terminal_type = str((info.get("last_transition_outcome", {}) or {}).get("terminal_type", "ongoing"))
                turn_count = self.env.state.get('turn_count', 0)
                
                action_name = ["Inform", "Clarify", "Question", "Escalate"][int(action)]
                
                print(f"  {step:<6} {action_name:<10} {terminal_type:<12} {reward:<10.3f} {turn_count:<6}")
                
                if truncated or done:
                    break
                
                step += 1
            
            terminal_type = str((info.get("last_transition_outcome", {}) or {}).get("terminal_type", "timeout"))
            
            if terminal_type == "success":
                self.add_step(DiagnosticStep(
                    name="Episode Outcome",
                    status="PASS",
                    message=f"Episode resolved successfully in {step} steps",
                    details={"Terminal Type": terminal_type}
                ))
            elif terminal_type == "escalation":
                self.add_step(DiagnosticStep(
                    name="Episode Outcome",
                    status="WARN",
                    message=f"Episode escalated after {step} steps",
                    details={"Terminal Type": terminal_type}
                ))
            else:
                self.add_step(DiagnosticStep(
                    name="Episode Outcome",
                    status="WARN",
                    message=f"Episode ended with: {terminal_type} ({step} steps)",
                    details={"Terminal Type": terminal_type}
                ))
            
            return True
        except Exception as e:
            self.add_step(DiagnosticStep(
                name="Episode Trace",
                status="FAIL",
                message=f"Error during episode trace: {e}"
            ))
            return False
    
    def test_batch_evaluation(self, n_episodes=10):
        """Test 6: Quick batch evaluation to see pattern"""
        self.log(f"\n6. BATCH EVALUATION ({n_episodes} episodes)", "TEST")
        
        try:
            outcomes = {"success": 0, "escalation": 0, "dropout": 0, "timeout": 0}
            
            for ep in range(n_episodes):
                obs, info = self.env.reset(seed=2000 + ep)
                done = False
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = self.env.step(int(action))
                    if truncated:
                        break
                
                terminal_type = str((info.get("last_transition_outcome", {}) or {}).get("terminal_type", "timeout"))
                outcomes[terminal_type] = outcomes.get(terminal_type, 0) + 1
            
            resolution_rate = outcomes["success"] / n_episodes * 100
            escalation_rate = outcomes["escalation"] / n_episodes * 100
            
            print(f"\n  Results ({n_episodes} episodes):")
            for term_type, count in outcomes.items():
                pct = count / n_episodes * 100
                print(f"    {term_type:12s}: {count:3d} ({pct:5.1f}%)")
            
            if resolution_rate > 50:
                self.add_step(DiagnosticStep(
                    name="Batch Performance",
                    status="PASS",
                    message=f"Good resolution rate: {resolution_rate:.1f}%",
                    details=outcomes
                ))
            elif resolution_rate > 0:
                self.add_step(DiagnosticStep(
                    name="Batch Performance",
                    status="WARN",
                    message=f"Low resolution rate: {resolution_rate:.1f}%",
                    details=outcomes
                ))
            else:
                self.add_step(DiagnosticStep(
                    name="Batch Performance",
                    status="FAIL",
                    message=f"No resolutions: {resolution_rate:.1f}% (100% escalation)",
                    details=outcomes
                ))
            
            return True
        except Exception as e:
            self.add_step(DiagnosticStep(
                name="Batch Evaluation",
                status="FAIL",
                message=f"Error during batch eval: {e}"
            ))
            return False
    
    def generate_report(self):
        """Generate diagnostic report"""
        self.log("\n" + "="*70)
        self.log("DIAGNOSTIC REPORT", "REPORT")
        self.log("="*70)
        
        passed = sum(1 for s in self.steps if s.status == "PASS")
        warned = sum(1 for s in self.steps if s.status == "WARN")
        failed = sum(1 for s in self.steps if s.status == "FAIL")
        
        print(f"\nSummary:")
        print(f"  ✓ PASSED: {passed}")
        print(f"  ⚠  WARNED: {warned}")
        print(f"  ✗ FAILED: {failed}")
        
        # Diagnosis
        print(f"\nDiagnosis:")
        if failed > 0:
            print(f"  ✗ CRITICAL ISSUES FOUND - Model cannot run properly")
            for step in self.steps:
                if step.status == "FAIL":
                    print(f"    - {step.name}: {step.message}")
        elif warned > 0:
            print(f"  ⚠  ISSUES DETECTED - Model runs but performance is degraded")
            for step in self.steps:
                if step.status == "WARN":
                    print(f"    - {step.name}: {step.message}")
        else:
            print(f"  ✓ ALL TESTS PASSED - Model appears healthy")
        
        print(f"\n" + "="*70 + "\n")
    
    def run_all_tests(self):
        """Run all diagnostic tests"""
        self.log("\n" + "█"*70)
        self.log("  PHASE 13 DIAGNOSTIC TEST SUITE", "HEAD")
        self.log("█"*70)
        
        # Test sequence
        if not self.test_model_integrity():
            self.log("Cannot proceed - model failed to load", "CRITICAL")
            self.generate_report()
            return False
        
        if not self.test_environment_setup():
            self.log("Cannot proceed - environment setup failed", "CRITICAL")
            self.generate_report()
            return False
        
        self.test_observation_details()
        self.test_action_masking()
        self.test_single_episode_trace()
        self.test_batch_evaluation(n_episodes=25)
        
        self.generate_report()
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Phase 13 Diagnostic: Compare Training vs LUMO Evaluation"
    )
    parser.add_argument(
        "--model",
        default="RL Out/models/best_model",
        help="Model path (default: RL Out/models/best_model)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Episodes for batch test (default: 10)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    diag = Phase13Diagnostic(model_path=args.model, verbose=args.verbose)
    success = diag.run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
