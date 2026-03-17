"""
AdaptiveBandit — Customer Support RL Agent
==========================================
Main entry point: run full experiment from data loading through evaluation.

Usage:
    python run_experiment.py                # Synthetic data, quick run
    python run_experiment.py --real-data    # Load real Twitter + OA data
    python run_experiment.py --gemini       # Enable Gemini enrichment
"""

import sys
import os
import argparse
import numpy as np

# Ensure package is importable
sys.path.insert(0, os.path.dirname(__file__))

from config import TRAINING_CONFIG, TIER_NAMES
from data.pipeline import build_unified_dataset
from data.feature_engineer import FEATURE_DIM
from environment.customer_env import CustomerSupportEnv
from agents.rule_based import RuleBasedAgent
from agents.linucb import LinUCBAgent
from agents.thompson_sampling import ThompsonSamplingAgent
from agents.dqn_agent import DQNAgent
from train import run_experiment, evaluate_on_tiers
from evaluation.metrics import (
    compute_business_metrics, compute_tier_stratified_metrics,
    compute_pareto_metrics, generate_comparison_table,
)
from evaluation.sensitivity import run_full_sensitivity
from evaluation.visualize import (
    plot_training_curves, plot_agent_comparison,
    plot_tier_performance, plot_pareto_frontier,
    plot_sensitivity_curves, plot_escalation_thresholds,
)
from rewards.economic_reward import get_all_thresholds


def main():
    parser = argparse.ArgumentParser(
        description="AdaptiveBandit: Customer Support RL Agent"
    )
    parser.add_argument('--real-data', action='store_true',
                        help='Load real Twitter + OpenAssistant data')
    parser.add_argument('--gemini', action='store_true',
                        help='Enable Gemini API enrichment')
    parser.add_argument('--episodes', type=int, default=3000,
                        help='Training episodes per agent')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip visualization')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ──────────────────────────────────────────────────
    # 1. Data
    # ──────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Building Dataset")
    print("=" * 60)

    train_convs, eval_convs = build_unified_dataset(
        use_twitter=args.real_data,
        use_openassistant=args.real_data,
        use_synthetic=True,
        seed=args.seed,
    )

    # Optional Gemini enrichment
    if args.gemini:
        try:
            from gemini_utils import enrich_conversations_with_gemini, gemini_available
            if gemini_available():
                print("\nEnriching with Gemini LLM-as-judge ...")
                train_convs = enrich_conversations_with_gemini(
                    train_convs, max_conversations=50
                )
            else:
                print("Gemini API not configured, skipping enrichment.")
        except Exception as e:
            print(f"Gemini enrichment failed: {e}")

    # ──────────────────────────────────────────────────
    # 2. Economic Thresholds
    # ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Economic Escalation Thresholds")
    print("=" * 60)
    thresholds = get_all_thresholds()
    for tier, t in thresholds.items():
        print(f"  {tier:12s}: Escalate if P(failure) > {t:.4f}")

    # ──────────────────────────────────────────────────
    # 3. Training
    # ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Training All Agents")
    print("=" * 60)

    results = run_experiment(
        conversations=train_convs,
        n_episodes=args.episodes,
        seed=args.seed,
        verbose=True,
    )

    # ──────────────────────────────────────────────────
    # 4. Evaluation
    # ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Business-Aligned Evaluation")
    print("=" * 60)

    # Comparison table
    print("\n" + generate_comparison_table(results))

    # Tier-stratified evaluation
    print("\n\nTier-Stratified Evaluation (LinUCB):")
    eval_env = CustomerSupportEnv(
        mode="bandit", conversations=eval_convs, seed=args.seed
    )
    tier_metrics = evaluate_on_tiers(
        results['agents']['linucb'], eval_env, n_eval=len(eval_convs)
    )
    for tier, m in tier_metrics.items():
        print(f"  {tier:12s} | "
              f"Reward: {m['avg_reward']:+.3f} | "
              f"Esc Rate: {m['escalation_rate']:.1%} | "
              f"Correct: {m['correct_routing_rate']:.1%}")

    # Pareto frontier
    pareto = compute_pareto_metrics(results)
    print("\nPareto Points (Cost vs CSAT):")
    for p in pareto:
        print(f"  {p['agent']:20s} | "
              f"Cost Saved: {p['cost_saved']:>8.1f} INR | "
              f"CSAT: {p['csat']:.2f}")

    # ──────────────────────────────────────────────────
    # 5. Sensitivity Analysis
    # ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Sensitivity Analysis")
    print("=" * 60)

    sensitivity = run_full_sensitivity()

    go_no_go = sensitivity['go_no_go']
    print(f"\nGo/No-Go Assessment:")
    print(f"  Annual savings:   INR {go_no_go['annual_savings']:>12,.0f}")
    print(f"  Total cost:       INR {go_no_go['total_cost']:>12,.0f}")
    print(f"  ROI:              {go_no_go['roi']:>12.1%}")
    print(f"  Payback:          {go_no_go['payback_months']:>12.1f} months")
    print(f"  Decision:         {'GO ✓' if go_no_go['go_decision'] else 'NO-GO ✗'}")

    # ──────────────────────────────────────────────────
    # 6. Visualization
    # ──────────────────────────────────────────────────
    if not args.no_plots:
        print("\n" + "=" * 60)
        print("STEP 6: Generating Visualizations")
        print("=" * 60)

        plot_training_curves(results)
        plot_agent_comparison(results)
        plot_escalation_thresholds()
        plot_pareto_frontier(pareto)
        plot_sensitivity_curves(sensitivity)

        if tier_metrics:
            plot_tier_performance(tier_metrics, agent_name="LinUCB")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
