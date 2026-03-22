"""
Sensitivity Analysis for Customer Support Bandit.

Analyzes how the optimal escalation policy changes under varying:
1. CSA cost (±20%)
2. Churn elasticity
3. LLM inference cost (±50%)
4. Capacity constraint K

Delivers sensitivity curves as specified in the proposal.
"""

import numpy as np
from typing import Dict, List, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import TIER_CONFIG, TIER_NAMES, BOT_INFERENCE_COST
from rewards.economic_reward import compute_escalation_threshold


def sensitivity_csa_cost(tier: str,
                          cost_range: Tuple[float, float] = (0.5, 2.0),
                          n_points: int = 20) -> Dict:
    """
    Analyze sensitivity of escalation threshold to CSA cost changes.

    Parameters
    ----------
    tier : str
        Customer tier.
    cost_range : tuple
        (min_multiplier, max_multiplier) for CSA cost.
    n_points : int
        Number of analysis points.

    Returns
    -------
    dict
        'multipliers': list of cost multipliers
        'thresholds': corresponding escalation thresholds
    """
    cfg = TIER_CONFIG[tier]
    base_cost = cfg['escalation_cost']
    multipliers = np.linspace(cost_range[0], cost_range[1], n_points)

    thresholds = []
    for m in multipliers:
        adjusted_cost = base_cost * m
        expected_churn_cost = cfg['churn_prob_bad_exp'] * cfg['clv']
        if expected_churn_cost > 0:
            threshold = adjusted_cost / expected_churn_cost
        else:
            threshold = 1.0
        thresholds.append(min(threshold, 1.0))

    return {
        'tier': tier,
        'parameter': 'csa_cost',
        'multipliers': multipliers.tolist(),
        'thresholds': thresholds,
        'base_threshold': compute_escalation_threshold(tier),
    }


def sensitivity_churn_elasticity(tier: str,
                                   elasticity_range: Tuple[float, float] = (0.01, 0.20),
                                   n_points: int = 20) -> Dict:
    """
    Analyze sensitivity to churn probability changes.

    Higher churn elasticity → lower threshold (more escalation).
    """
    cfg = TIER_CONFIG[tier]
    churn_probs = np.linspace(elasticity_range[0], elasticity_range[1], n_points)

    thresholds = []
    monthly_impact = []

    for churn_p in churn_probs:
        expected_churn_cost = churn_p * cfg['clv']
        if expected_churn_cost > 0:
            threshold = cfg['escalation_cost'] / expected_churn_cost
        else:
            threshold = 1.0
        thresholds.append(min(threshold, 1.0))
        monthly_impact.append(expected_churn_cost / 12.0)

    return {
        'tier': tier,
        'parameter': 'churn_elasticity',
        'churn_probabilities': churn_probs.tolist(),
        'thresholds': thresholds,
        'monthly_churn_impact': monthly_impact,
        'base_churn_prob': cfg['churn_prob_bad_exp'],
    }


def sensitivity_inference_cost(cost_ratio_range: Tuple[float, float] = (10, 100),
                                 n_points: int = 20) -> Dict:
    """
    Analyze sensitivity to the CSA/inference cost ratio.

    C_CSA / C_inf ∈ [10, 100]
    Higher ratio → more aggressive bot usage is profitable.
    """
    ratios = np.linspace(cost_ratio_range[0], cost_ratio_range[1], n_points)

    optimal_deflection = []
    for ratio in ratios:
        # Higher ratio → bots are relatively cheaper → higher deflection optimal
        # Simplified model: deflection_rate = 1 - 1/sqrt(ratio)
        deflection = max(0, 1.0 - 1.0 / np.sqrt(ratio))
        optimal_deflection.append(deflection)

    return {
        'parameter': 'cost_ratio',
        'cost_ratios': ratios.tolist(),
        'optimal_deflection_rates': optimal_deflection,
        'current_ratio': TIER_CONFIG['Business+']['escalation_cost'] / BOT_INFERENCE_COST,
    }


def sensitivity_capacity_k(tier: str,
                             k_range: Tuple[int, int] = (10, 200),
                             n_conversations: int = 1000,
                             n_points: int = 20) -> Dict:
    """
    Analyze how capacity constraint K affects policy.

    More capacity → more escalations possible → lower threshold.
    """
    k_values = np.linspace(k_range[0], k_range[1], n_points).astype(int)
    cfg = TIER_CONFIG[tier]

    results = []
    for k in k_values:
        # Max escalation rate given capacity
        max_esc_rate = k / n_conversations
        # Effective threshold with capacity constraint
        base_threshold = compute_escalation_threshold(tier)
        effective_threshold = max(base_threshold, 1.0 - max_esc_rate)
        results.append({
            'k': int(k),
            'max_esc_rate': max_esc_rate,
            'effective_threshold': effective_threshold,
        })

    return {
        'tier': tier,
        'parameter': 'capacity_k',
        'results': results,
    }


def run_full_sensitivity(tiers: List[str] = None) -> Dict:
    """
    Run comprehensive sensitivity analysis across all parameters and tiers.

    Returns
    -------
    dict
        Complete sensitivity analysis results.
    """
    if tiers is None:
        tiers = ['Pro', 'Business+', 'Enterprise']

    analysis = {
        'csa_cost': {},
        'churn_elasticity': {},
        'inference_cost': sensitivity_inference_cost(),
        'capacity': {},
    }

    for tier in tiers:
        analysis['csa_cost'][tier] = sensitivity_csa_cost(tier)
        analysis['churn_elasticity'][tier] = sensitivity_churn_elasticity(tier)
        analysis['capacity'][tier] = sensitivity_capacity_k(tier)

    # Go/No-Go evaluation
    analysis['go_no_go'] = evaluate_go_no_go()

    return analysis


def evaluate_go_no_go(monthly_users: int = 10000,
                       ml_development_cost: float = 500000,
                       annual_infra_cost: float = 120000) -> Dict:
    """
    Evaluate Go/No-Go deployment rule:
    If (CostSaved/user * Users * 12) > (MLCost + InfCost) and ΔCSAT > -0.1

    Parameters
    ----------
    monthly_users : int
        Total monthly active users.
    ml_development_cost : float
        One-time ML development cost (INR).
    annual_infra_cost : float
        Annual infrastructure cost (INR).

    Returns
    -------
    dict
        Go/No-Go analysis.
    """
    # Estimate cost savings per user per month (conservative)
    tier_mix = {'Free': 0.60, 'Pro': 0.25, 'Business+': 0.12, 'Enterprise': 0.03}

    total_savings = 0
    for tier, frac in tier_mix.items():
        cfg = TIER_CONFIG[tier]
        # Assume 20% improvement in routing accuracy
        savings = cfg['escalation_cost'] * 0.20 * 0.5  # 50% of conversations need decision
        total_savings += frac * savings

    annual_savings = total_savings * monthly_users * 12
    total_cost = ml_development_cost + annual_infra_cost

    roi = (annual_savings - total_cost) / max(total_cost, 1)
    payback_months = total_cost / max(total_savings * monthly_users, 1)

    return {
        'estimated_savings_per_user_month': total_savings,
        'annual_savings': annual_savings,
        'total_cost': total_cost,
        'roi': roi,
        'payback_months': payback_months,
        'go_decision': annual_savings > total_cost,
        'breakeven_users': int(total_cost / max(total_savings * 12, 1)),
    }
