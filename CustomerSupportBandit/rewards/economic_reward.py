"""
Economic Reward Function for Customer Support.

Implements the tier-specific economic reward as defined in the AML proposal:

    R_t = +alpha * 1{Resolved}
          - C_escalation * 1{Escalate}
          - L_churn * 1{Failure}
          - delta * TurnPenalty

Economically optimal escalation threshold:
    Escalate if p * Expected_Churn_Cost > Escalation_Cost

where p is the estimated probability of automated resolution failure.
"""

import numpy as np
from typing import Dict, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import TIER_CONFIG, TIER_NAMES, REWARD_WEIGHTS, BOT_INFERENCE_COST


def compute_escalation_threshold(tier: str) -> float:
    """
    Compute the economically optimal escalation threshold for a tier.

    Escalate if:  p > C_escalation / (churn_prob * CLV)

    Parameters
    ----------
    tier : str
        Customer tier name.

    Returns
    -------
    float
        Threshold probability. If estimated failure probability exceeds
        this, the economically optimal action is to escalate.
    """
    cfg = TIER_CONFIG[tier]
    escalation_cost = cfg['escalation_cost']
    expected_churn_cost = cfg['churn_prob_bad_exp'] * cfg['clv']

    if expected_churn_cost <= 0:
        return 1.0  # Never escalate (e.g., Free tier with no CLV)

    threshold = escalation_cost / expected_churn_cost
    return min(threshold, 1.0)


def get_all_thresholds() -> Dict[str, float]:
    """Compute escalation thresholds for all tiers."""
    return {tier: compute_escalation_threshold(tier) for tier in TIER_NAMES}


def compute_economic_reward(action: str, tier: str,
                             resolved: bool,
                             turn_count: int,
                             satisfaction_score: float = 0.5) -> Tuple[float, Dict]:
    """
    Compute the full economic reward for a support interaction.

    Parameters
    ----------
    action : str
        Terminal action taken ('bot', 'human', 'close', 'escalate').
    tier : str
        Customer tier.
    resolved : bool
        Whether the issue was resolved.
    turn_count : int
        Number of turns in the conversation.
    satisfaction_score : float
        Estimated customer satisfaction [0, 1].

    Returns
    -------
    reward : float
        Total economic reward.
    breakdown : dict
        Component-level reward breakdown.
    """
    cfg = TIER_CONFIG[tier]
    w = REWARD_WEIGHTS

    breakdown = {}

    # Resolution bonus/penalty
    if resolved:
        resolution_reward = w['resolution_bonus']
    else:
        churn_cost = cfg['churn_prob_bad_exp'] * cfg['clv']
        resolution_reward = -w['churn_penalty_scale'] * churn_cost / 1000.0
    breakdown['resolution'] = resolution_reward

    # Escalation cost (only if human used)
    if action in ('human', 'escalate'):
        esc_cost = -w['escalation_penalty_scale'] * cfg['escalation_cost'] / 50.0
    else:
        esc_cost = w['gamma_cost_saved'] * cfg['escalation_cost'] / 50.0
    breakdown['escalation_cost'] = esc_cost

    # Turn penalty
    turn_penalty = -w['turn_penalty'] * turn_count
    breakdown['turn_penalty'] = turn_penalty

    # Satisfaction component
    sat_reward = w['alpha_sentiment'] * (satisfaction_score - 0.5) * 2.0
    breakdown['satisfaction'] = sat_reward

    # Inference cost (bot path only)
    if action in ('bot', 'close'):
        inf_cost = -BOT_INFERENCE_COST / 10.0
    else:
        inf_cost = 0.0
    breakdown['inference_cost'] = inf_cost

    total = resolution_reward + esc_cost + turn_penalty + sat_reward + inf_cost
    breakdown['total'] = total

    return total, breakdown


def compute_monthly_profit(tier: str,
                            n_conversations: int,
                            n_escalated: int,
                            n_churned: int,
                            n_resolved: int) -> Dict:
    """
    Compute expected monthly profit for a tier under a given policy.

    Profit = Base_Revenue - C_escalation * Escalations
             - L_churn * ChurnEvents - InferenceCost

    Parameters
    ----------
    tier : str
        Customer tier.
    n_conversations : int
        Total monthly conversations.
    n_escalated : int
        Number escalated to human.
    n_churned : int
        Number of customers who churned.
    n_resolved : int
        Number successfully resolved.

    Returns
    -------
    dict
        Profit breakdown.
    """
    cfg = TIER_CONFIG[tier]

    base_revenue = cfg['monthly_profit'] * n_conversations
    escalation_costs = cfg['escalation_cost'] * n_escalated
    churn_costs = cfg['clv'] * n_churned / 12.0  # Monthly fraction of CLV loss
    inference_costs = BOT_INFERENCE_COST * (n_conversations - n_escalated)
    resolution_rate = n_resolved / max(n_conversations, 1)

    profit = base_revenue - escalation_costs - churn_costs - inference_costs

    return {
        'tier': tier,
        'base_revenue': base_revenue,
        'escalation_costs': escalation_costs,
        'churn_costs': churn_costs,
        'inference_costs': inference_costs,
        'net_profit': profit,
        'cost_per_resolution': (escalation_costs + inference_costs) / max(n_resolved, 1),
        'resolution_rate': resolution_rate,
        'deflection_rate': 1.0 - n_escalated / max(n_conversations, 1),
        'churn_rate': n_churned / max(n_conversations, 1),
    }


def compute_cost_saved(baseline_profit: Dict, policy_profit: Dict) -> Dict:
    """
    Compute cost savings of a policy vs baseline.

    CostSaved = (delta * C_esc * V) - (C_inf * V) - C_churn
    """
    return {
        'cost_reduction': policy_profit['net_profit'] - baseline_profit['net_profit'],
        'escalation_savings': baseline_profit['escalation_costs'] - policy_profit['escalation_costs'],
        'churn_savings': baseline_profit['churn_costs'] - policy_profit['churn_costs'],
        'cost_per_resolution_improvement': (
            baseline_profit['cost_per_resolution'] - policy_profit['cost_per_resolution']
        ),
        'deflection_rate_change': (
            policy_profit['deflection_rate'] - baseline_profit['deflection_rate']
        ),
    }


def compute_priority_score(escalation_risk: float,
                            clv: float,
                            interaction_cost: float,
                            csat: float) -> float:
    """
    Compute capacity allocation priority score.

    Priority = EscRisk * (CLV - C_int) * (1 - CSAT/5)

    Top-K conversations by priority get human agents.
    """
    return escalation_risk * (clv - interaction_cost) * (1.0 - csat / 5.0)
