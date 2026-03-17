"""
Visualization utilities for Customer Support Bandit.

Generates all plots specified in the AML proposal:
- Training curves
- Tier-stratified performance
- Pareto frontier (Cost Saved vs CSAT)
- Sensitivity curves
- Economic impact analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import TIER_NAMES, TIER_CONFIG, BANDIT_ACTIONS


# Style settings
COLORS = {
    'rule_based': '#95a5a6',
    'linucb': '#e74c3c',
    'thompson_sampling': '#3498db',
    'dqn': '#2ecc71',
}

TIER_COLORS = {
    'Free': '#bdc3c7',
    'Pro': '#3498db',
    'Business+': '#e67e22',
    'Enterprise': '#e74c3c',
}


def plot_training_curves(results: Dict, save_path: Optional[str] = None):
    """Plot cumulative reward curves for all agents."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Evaluation reward over episodes
    ax = axes[0]
    for name in ['rule_based', 'linucb', 'thompson_sampling']:
        if name not in results:
            continue
        h = results[name]
        if 'eval_points' in h and 'eval_rewards' in h:
            ax.plot(h['eval_points'], h['eval_rewards'],
                    label=name, color=COLORS.get(name, 'gray'), linewidth=2)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward (per 100 episodes)')
    ax.set_title('Phase I: Contextual Bandit Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: DQN training
    ax = axes[1]
    if 'dqn' in results:
        h = results['dqn']
        if 'eval_points' in h and 'eval_rewards' in h:
            ax.plot(h['eval_points'], h['eval_rewards'],
                    color=COLORS['dqn'], linewidth=2, label='DQN')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Episode Reward')
        ax.set_title('Phase II: DQN (MDP) Training')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add epsilon on secondary axis
        if 'epsilon_history' in h:
            ax2 = ax.twinx()
            ax2.plot(h['epsilon_history'], color='gray', alpha=0.3,
                     linewidth=1, label='ε')
            ax2.set_ylabel('Epsilon', color='gray')
            ax2.legend(loc='upper right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_agent_comparison(results: Dict, save_path: Optional[str] = None):
    """Bar chart comparing key metrics across agents."""
    from evaluation.metrics import compute_business_metrics

    agents = ['rule_based', 'linucb', 'thompson_sampling']
    agents = [a for a in agents if a in results]

    metrics_list = [compute_business_metrics(results[a]) for a in agents]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Avg Reward
    ax = axes[0]
    values = [m['avg_reward'] for m in metrics_list]
    bars = ax.bar(agents, values, color=[COLORS.get(a, 'gray') for a in agents])
    ax.set_ylabel('Average Reward')
    ax.set_title('Average Reward per Conversation')
    ax.set_xticklabels(agents, rotation=15)

    # Correct Routing Rate
    ax = axes[1]
    values = [m['correct_routing_rate'] for m in metrics_list]
    ax.bar(agents, values, color=[COLORS.get(a, 'gray') for a in agents])
    ax.set_ylabel('Correct Routing Rate')
    ax.set_title('Routing Accuracy')
    ax.set_xticklabels(agents, rotation=15)
    ax.set_ylim(0, 1)

    # Escalation Rate
    ax = axes[2]
    values = [m['escalation_rate'] for m in metrics_list]
    ax.bar(agents, values, color=[COLORS.get(a, 'gray') for a in agents])
    ax.set_ylabel('Escalation Rate')
    ax.set_title('Human Escalation Rate')
    ax.set_xticklabels(agents, rotation=15)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_tier_performance(tier_metrics: Dict, agent_name: str = "",
                           save_path: Optional[str] = None):
    """Plot performance breakdown by tier."""
    tiers = [t for t in TIER_NAMES if t in tier_metrics]
    if not tiers:
        print("No tier data available.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Per-tier avg reward
    ax = axes[0]
    values = [tier_metrics[t]['avg_reward'] for t in tiers]
    ax.bar(tiers, values, color=[TIER_COLORS.get(t, 'gray') for t in tiers])
    ax.set_ylabel('Average Reward')
    ax.set_title(f'{agent_name}: Reward by Tier')

    # Per-tier escalation rate
    ax = axes[1]
    esc_rates = [tier_metrics[t]['escalation_rate'] for t in tiers]
    thresholds = [tier_metrics[t].get('threshold', 1.0) for t in tiers]
    x = np.arange(len(tiers))
    ax.bar(x, esc_rates, color=[TIER_COLORS.get(t, 'gray') for t in tiers],
           label='Actual')
    ax.scatter(x, thresholds, color='red', zorder=5, s=100, marker='_',
              linewidths=3, label='Max Threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel('Escalation Rate')
    ax.set_title(f'{agent_name}: Escalation Rate by Tier')
    ax.legend()

    # Per-tier correct routing
    ax = axes[2]
    values = [tier_metrics[t]['correct_routing_rate'] for t in tiers]
    ax.bar(tiers, values, color=[TIER_COLORS.get(t, 'gray') for t in tiers])
    ax.set_ylabel('Correct Routing Rate')
    ax.set_title(f'{agent_name}: Routing Accuracy by Tier')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_pareto_frontier(pareto_points: List[Dict],
                          save_path: Optional[str] = None):
    """
    Plot the (Cost Saved, CSAT) Pareto frontier.

    The decision-maker selects the operating point on this frontier
    based on their tier mix and budget.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for point in pareto_points:
        color = COLORS.get(point['agent'], 'gray')
        ax.scatter(point['cost_saved'], point['csat'],
                   color=color, s=200, zorder=5, edgecolors='black')
        ax.annotate(point['agent'],
                   (point['cost_saved'], point['csat']),
                   textcoords="offset points",
                   xytext=(10, 10), fontsize=10)

    # CSAT floor line
    ax.axhline(y=3.5, color='red', linestyle='--', alpha=0.5,
               label='CSAT Floor (3.5)')

    ax.set_xlabel('Estimated Cost Saved (INR / conversation)')
    ax.set_ylabel('Estimated CSAT Score')
    ax.set_title('Pareto Frontier: Cost Saved vs. CSAT')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_sensitivity_curves(sensitivity_results: Dict,
                             save_path: Optional[str] = None):
    """Plot sensitivity analysis curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. CSA Cost sensitivity
    ax = axes[0, 0]
    csa_data = sensitivity_results.get('csa_cost', {})
    for tier, data in csa_data.items():
        ax.plot(data['multipliers'], data['thresholds'],
                label=tier, color=TIER_COLORS.get(tier, 'gray'), linewidth=2)
    ax.set_xlabel('CSA Cost Multiplier')
    ax.set_ylabel('Optimal Escalation Threshold')
    ax.set_title('Sensitivity: CSA Cost → Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Churn Elasticity
    ax = axes[0, 1]
    churn_data = sensitivity_results.get('churn_elasticity', {})
    for tier, data in churn_data.items():
        ax.plot(data['churn_probabilities'], data['thresholds'],
                label=tier, color=TIER_COLORS.get(tier, 'gray'), linewidth=2)
        ax.axvline(x=data['base_churn_prob'], color=TIER_COLORS.get(tier, 'gray'),
                   linestyle='--', alpha=0.3)
    ax.set_xlabel('Churn Probability (given bad experience)')
    ax.set_ylabel('Optimal Escalation Threshold')
    ax.set_title('Sensitivity: Churn Elasticity → Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Inference Cost Ratio
    ax = axes[1, 0]
    inf_data = sensitivity_results.get('inference_cost', {})
    if inf_data:
        ax.plot(inf_data['cost_ratios'], inf_data['optimal_deflection_rates'],
                color='#2ecc71', linewidth=2)
        ax.axvline(x=inf_data['current_ratio'], color='red',
                   linestyle='--', alpha=0.5, label='Current ratio')
    ax.set_xlabel('Cost Ratio (C_CSA / C_inference)')
    ax.set_ylabel('Optimal Deflection Rate')
    ax.set_title('Sensitivity: Cost Ratio → Deflection Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Go/No-Go
    ax = axes[1, 1]
    go_data = sensitivity_results.get('go_no_go', {})
    if go_data:
        labels = ['Annual\nSavings', 'Total\nCost']
        values = [go_data.get('annual_savings', 0) / 1e6,
                  go_data.get('total_cost', 0) / 1e6]
        colors = ['#2ecc71', '#e74c3c']
        bars = ax.bar(labels, values, color=colors)
        ax.set_ylabel('Amount (₹ Millions)')
        ax.set_title(f'Go/No-Go: ROI = {go_data.get("roi", 0):.1%}')

        # Add text
        decision = "GO ✓" if go_data.get('go_decision', False) else "NO-GO ✗"
        ax.text(0.5, 0.9, decision, transform=ax.transAxes,
                fontsize=20, ha='center', fontweight='bold',
                color='green' if go_data.get('go_decision', False) else 'red')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_escalation_thresholds(save_path: Optional[str] = None):
    """Plot economically optimal escalation thresholds by tier."""
    from rewards.economic_reward import get_all_thresholds

    thresholds = get_all_thresholds()
    tiers = list(thresholds.keys())
    values = list(thresholds.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(tiers, values,
                  color=[TIER_COLORS.get(t, 'gray') for t in tiers])

    # Annotate
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12)

    ax.set_ylabel('Failure Probability Threshold')
    ax.set_title('Economically Optimal Escalation Thresholds by Tier\n'
                 '(Escalate if P(failure) > threshold)')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
