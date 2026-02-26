"""
Evaluation Metrics for Customer Support Bandit.

Business-aligned metrics as specified in the AML proposal:
- Expected monthly cost reduction
- Tier-specific escalation rates
- CSAT impact
- Churn-adjusted profit
- Capacity utilization
"""

import numpy as np
from typing import Dict, List

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import TIER_CONFIG, TIER_NAMES, BOT_INFERENCE_COST
from rewards.economic_reward import compute_monthly_profit, compute_cost_saved


def compute_business_metrics(history: Dict, tier_label: str = "all") -> Dict:
    """
    Compute business-aligned metrics from training history.

    Parameters
    ----------
    history : dict
        Training history from train_bandit or train_dqn.
    tier_label : str
        Which tier to compute for (or "all").

    Returns
    -------
    dict
        Business metrics.
    """
    rewards = np.array(history['rewards'])
    actions = np.array(history['actions'])
    outcomes = history.get('outcomes', [])

    n = len(rewards)
    n_escalated = np.sum(actions == 1)
    n_bot = np.sum(actions == 0)

    # Outcome analysis
    correct_deflections = sum(1 for o in outcomes if o == 'correct_deflection')
    correct_escalations = sum(1 for o in outcomes if o == 'correct_escalation')
    missed_escalations = sum(1 for o in outcomes if o == 'missed_escalation')
    unnecessary_escalations = sum(1 for o in outcomes if o == 'unnecessary_escalation')

    metrics = {
        'total_conversations': n,
        'total_reward': float(np.sum(rewards)),
        'avg_reward': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),

        # Routing
        'escalation_rate': float(n_escalated / max(n, 1)),
        'deflection_rate': float(n_bot / max(n, 1)),

        # Accuracy
        'correct_routing_rate': float(
            (correct_deflections + correct_escalations) / max(n, 1)
        ),
        'correct_deflection_rate': float(
            correct_deflections / max(n_bot, 1)
        ),
        'correct_escalation_rate': float(
            correct_escalations / max(n_escalated, 1)
        ),

        # Errors
        'missed_escalation_rate': float(
            missed_escalations / max(n, 1)
        ),
        'unnecessary_escalation_rate': float(
            unnecessary_escalations / max(n, 1)
        ),

        # Economic
        'estimated_cost_savings_per_conv': float(
            np.mean(rewards) * 50  # Scale to INR equivalent
        ),
    }

    return metrics


def compute_tier_stratified_metrics(history: Dict) -> Dict:
    """
    Compute metrics stratified by customer tier.

    Returns per-tier performance breakdown.
    """
    tier_metrics = {}
    tiers = history.get('tiers', [])
    rewards = np.array(history['rewards'])
    actions = np.array(history['actions'])
    outcomes = history.get('outcomes', [])

    for tier_name in TIER_NAMES:
        mask = [t == tier_name for t in tiers]
        if sum(mask) == 0:
            continue

        tier_rewards = rewards[mask]
        tier_actions = actions[mask]
        tier_outcomes = [o for o, m in zip(outcomes, mask) if m]

        n = len(tier_rewards)
        n_esc = np.sum(tier_actions == 1)

        correct = sum(1 for o in tier_outcomes
                      if o.startswith('correct_'))

        tier_metrics[tier_name] = {
            'n_conversations': n,
            'avg_reward': float(np.mean(tier_rewards)),
            'escalation_rate': float(n_esc / max(n, 1)),
            'correct_routing_rate': float(correct / max(n, 1)),

            # Compare to threshold
            'threshold': TIER_CONFIG[tier_name].get('max_escalation_rate', 1.0),
            'within_threshold': n_esc / max(n, 1) <= TIER_CONFIG[tier_name].get('max_escalation_rate', 1.0),
        }

    return tier_metrics


def compute_pareto_metrics(results: Dict) -> List[Dict]:
    """
    Compute (Cost Saved, CSAT) points for Pareto frontier analysis.

    Each agent's operating point on the Pareto frontier.
    """
    pareto_points = []

    for agent_name, history in results.items():
        if agent_name == 'agents':
            continue

        metrics = compute_business_metrics(history)

        # Proxy CSAT from correct routing
        csat_proxy = 3.0 + 2.0 * metrics['correct_routing_rate']

        pareto_points.append({
            'agent': agent_name,
            'cost_saved': metrics['estimated_cost_savings_per_conv'],
            'csat': csat_proxy,
            'escalation_rate': metrics['escalation_rate'],
            'correct_routing': metrics['correct_routing_rate'],
        })

    return pareto_points


def generate_comparison_table(results: Dict) -> str:
    """
    Generate a formatted comparison table across all agents.

    Returns a string suitable for printing or inclusion in reports.
    """
    header = (f"{'Agent':<20} {'Avg Reward':>10} {'Esc Rate':>10} "
              f"{'Correct %':>10} {'Missed':>10} {'Unnecessary':>12}")
    separator = "-" * len(header)

    rows = [header, separator]

    for agent_name in ['rule_based', 'linucb', 'thompson_sampling', 'dqn']:
        if agent_name not in results:
            continue
        metrics = compute_business_metrics(results[agent_name])

        row = (f"{agent_name:<20} "
               f"{metrics['avg_reward']:>+10.3f} "
               f"{metrics['escalation_rate']:>10.1%} "
               f"{metrics['correct_routing_rate']:>10.1%} "
               f"{metrics['missed_escalation_rate']:>10.1%} "
               f"{metrics['unnecessary_escalation_rate']:>12.1%}")
        rows.append(row)

    return "\n".join(rows)
