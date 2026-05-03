#!/usr/bin/env python3
"""
Phase 13 Full LUMO-Based Evaluation
====================================

Comprehensive evaluation of Phase 13 model on LUMO-based realistic scenarios.

Tracks:
  - Resolution rate (PRIMARY METRIC) ✓
  - Escalation rate
  - Dropout rate
  - Timeout rate
  - Average turns to resolution
  - Cumulative reward distribution
  - Customer satisfaction proxy (resolution quality)
  - Performance by tier (Free/Pro/Business/Enterprise)
  - Performance by persona

Run: python phase13_lumo_evaluation.py [--episodes 100] [--output results.json]

Estimated runtime:
  - 25 episodes: ~2-5 minutes
  - 100 episodes: ~8-20 minutes (depending on LLM backend)
  - 250+ episodes: ~20-60 minutes
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

PLOTTING_AVAILABLE = True
plt = None
sns = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception as plotting_import_error:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None
    print(f"⚠ Plotting disabled: {plotting_import_error}")

from Simulation_4.env.support_env import SupportEnv
from Simulation_4.rag.scenario_generator import ScenarioGenerator
from Simulation_4.training.action_masking import ActionMaskedEnv
from Simulation_4.training.nlp_observation import NLPObservationWrapper
from Simulation_4.env.agent_response_generator import AgentResponseGenerator

if PLOTTING_AVAILABLE:
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class EpisodeMetrics:
    episode_id: int
    reward: float
    turns: int
    terminal_type: str  # success, escalation, dropout, timeout
    tier: str
    persona: str
    satisfaction_proxy: float  # 0-1 based on resolution quality


@dataclass
class AggregateMetrics:
    total_episodes: int
    resolution_rate: float
    escalation_rate: float
    dropout_rate: float
    timeout_rate: float
    
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    
    mean_turns_to_resolution: float
    std_turns_to_resolution: float
    
    mean_satisfaction: float
    
    # By tier
    resolution_by_tier: dict = field(default_factory=dict)
    escalation_by_tier: dict = field(default_factory=dict)
    
    # By persona
    resolution_by_persona: dict = field(default_factory=dict)
    escalation_by_persona: dict = field(default_factory=dict)
    
    # Cumulative reward distribution
    reward_distribution: dict = field(default_factory=dict)


# ============================================================================
# Evaluation
# ============================================================================

def calculate_satisfaction_proxy(terminal_type: str, turns: int, reward: float) -> float:
    """
    Proxy for customer satisfaction:
    - Success = high satisfaction (1.0)
    - Escalation = medium (0.6 - penalize for not solving)
    - Dropout/timeout = low (0.1 - customer left)
    
    Adjust by efficiency (fewer turns = better)
    """
    base_scores = {
        "success": 1.0,
        "escalation": 0.65,
        "dropout": 0.15,
        "timeout": 0.10,
    }
    base = base_scores.get(terminal_type, 0.0)
    
    # Efficiency bonus/penalty: ideal is 2-4 turns
    turns_penalty = max(0, (turns - 4) * 0.02)  # -2% per extra turn after 4
    turns_bonus = max(0, (4 - turns) * 0.05) if turns < 4 else 0  # +5% per turn saved
    
    efficiency_adjustment = turns_bonus - turns_penalty
    return min(1.0, max(0.0, base + efficiency_adjustment))


def evaluate_phase13_lumo(
    model_path: str,
    artifacts_root: str,
    n_episodes: int = 100,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Full evaluation of Phase 13 model on LUMO scenarios.
    """
    
    print("\n" + "█" * 70)
    print("  PHASE 13 LUMO EVALUATION")
    print("█" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {model_path}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Artifacts root: {artifacts_root}")
    print()
    
    # Load model
    print("Loading model...")
    start_time = time.time()
    model = PPO.load(model_path)
    print(f"✓ Model loaded ({time.time() - start_time:.2f}s)\n")
    
    # Create environment WITH NLP PROCESSING (same as training)
    print("Initializing environment with NLP observation wrapper...")
    env = SupportEnv(artifacts_root=artifacts_root, nlg_enabled=True)  # NLG enabled for realistic eval
    
    agent_gen = AgentResponseGenerator(model="llama3")
    env = NLPObservationWrapper(                                       # NLP wrapper - CRITICAL!
        env,
        intent_model="phi3",
        agent_response_generator=agent_gen,
    )
    env = ActionMaskedEnv(env)  # Wrap with ActionMaskedEnv (same as training)
    print(f"✓ Environment ready with NLP pipeline\n")
    
    # Initialize scenario generator
    scenario_gen = ScenarioGenerator(
        templates_path=str(Path(artifacts_root) / ".." / "rag" / "scenario_templates.json"),
        subflow_mapping_path=str(Path(artifacts_root) / ".." / "rag" / "subflow_mapping.json"),
    )
    
    # Track results
    episode_metrics: list[EpisodeMetrics] = []
    rewards = []
    turns_to_resolution = []
    
    tier_counts = {"Free": 0, "Pro": 0, "Business": 0, "Enterprise": 0}
    tier_resolutions = tier_counts.copy()
    tier_escalations = tier_counts.copy()
    
    persona_counts = {
        "high_engagement_resolver": 0,
        "low_engagement_resolver": 0,
        "silent_dropout": 0,
        "escalation_prone": 0,
    }
    persona_resolutions = persona_counts.copy()
    persona_escalations = persona_counts.copy()
    
    satisfactions = []
    
    # ========================================================================
    # Episode Loop
    # ========================================================================
    
    print(f"Running {n_episodes} episodes...\n")
    eval_start = time.time()
    
    for ep in range(n_episodes):
        if (ep + 1) % max(1, n_episodes // 10) == 0:
            elapsed = time.time() - eval_start
            rate = (ep + 1) / elapsed
            eta = (n_episodes - ep - 1) / rate if rate > 0 else 0
            print(f"  [{ep+1}/{n_episodes}] Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
        
        # Reset environment
        obs, _ = env.reset(seed=50_000 + ep)
        done = False
        ep_reward = 0.0
        turns = 0
        info = {}
        
        # Run episode
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            ep_reward += float(reward)
            turns += 1
            
            if truncated:
                break
        
        # Extract metrics
        terminal_type = str(
            (info.get("last_transition_outcome", {}) or {}).get("terminal_type", "timeout")
        )
        tier = str(env.unwrapped.state.get("tier", "Free"))
        persona = str(env.unwrapped.state.get("persona_label", "unknown"))
        
        # Calculate satisfaction proxy
        satisfaction = calculate_satisfaction_proxy(terminal_type, turns, ep_reward)
        
        # Store episode metrics
        episode_metrics.append(
            EpisodeMetrics(
                episode_id=ep,
                reward=ep_reward,
                turns=turns,
                terminal_type=terminal_type,
                tier=tier,
                persona=persona,
                satisfaction_proxy=satisfaction,
            )
        )
        
        rewards.append(ep_reward)
        satisfactions.append(satisfaction)
        
        # Update counters
        if tier in tier_counts:
            tier_counts[tier] += 1
            if terminal_type == "success":
                tier_resolutions[tier] += 1
            elif terminal_type == "escalation":
                tier_escalations[tier] += 1
        
        if persona in persona_counts:
            persona_counts[persona] += 1
            if terminal_type == "success":
                persona_resolutions[persona] += 1
            elif terminal_type == "escalation":
                persona_escalations[persona] += 1
        
        if terminal_type == "success":
            turns_to_resolution.append(turns)
    
    eval_duration = time.time() - eval_start
    
    # ========================================================================
    # Aggregate Results
    # ========================================================================
    
    print(f"\n✓ Evaluation complete ({eval_duration:.1f}s)\n")
    
    resolution_count = sum(1 for m in episode_metrics if m.terminal_type == "success")
    escalation_count = sum(1 for m in episode_metrics if m.terminal_type == "escalation")
    dropout_count = sum(1 for m in episode_metrics if m.terminal_type == "dropout")
    timeout_count = sum(1 for m in episode_metrics if m.terminal_type == "timeout")
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": {
            "model_path": model_path,
            "n_episodes": n_episodes,
            "artifacts_root": artifacts_root,
        },
        "timing": {
            "duration_seconds": eval_duration,
            "episodes_per_minute": (n_episodes / eval_duration) * 60,
        },
        # ====================================================================
        # PRIMARY METRIC: RESOLUTION RATE
        # ====================================================================
        "resolution_rate": float(resolution_count / n_episodes),
        "escalation_rate": float(escalation_count / n_episodes),
        "dropout_rate": float(dropout_count / n_episodes),
        "timeout_rate": float(timeout_count / n_episodes),
        
        # ====================================================================
        # TERMINAL TYPE COUNTS
        # ====================================================================
        "terminal_counts": {
            "success": resolution_count,
            "escalation": escalation_count,
            "dropout": dropout_count,
            "timeout": timeout_count,
        },
        
        # ====================================================================
        # REWARD METRICS
        # ====================================================================
        "reward": {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
            "median": float(np.median(rewards)),
        },
        
        # ====================================================================
        # TURNS TO RESOLUTION
        # ====================================================================
        "turns_analysis": {
            "mean_turns_to_resolution": float(np.mean(turns_to_resolution)) if turns_to_resolution else 0.0,
            "std_turns_to_resolution": float(np.std(turns_to_resolution)) if turns_to_resolution else 0.0,
            "min_turns": int(np.min(turns_to_resolution)) if turns_to_resolution else 0,
            "max_turns": int(np.max(turns_to_resolution)) if turns_to_resolution else 0,
        },
        
        # ====================================================================
        # CUSTOMER SATISFACTION (PROXY)
        # ====================================================================
        "satisfaction": {
            "mean_satisfaction_proxy": float(np.mean(satisfactions)),
            "std_satisfaction_proxy": float(np.std(satisfactions)),
        },
        
        # ====================================================================
        # BY TIER ANALYSIS
        # ====================================================================
        "by_tier": {
            tier: {
                "count": tier_counts[tier],
                "resolution_rate": float(tier_resolutions[tier] / tier_counts[tier]) if tier_counts[tier] > 0 else 0.0,
                "escalation_rate": float(tier_escalations[tier] / tier_counts[tier]) if tier_counts[tier] > 0 else 0.0,
            }
            for tier in tier_counts
        },
        
        # ====================================================================
        # BY PERSONA ANALYSIS
        # ====================================================================
        "by_persona": {
            persona: {
                "count": persona_counts[persona],
                "resolution_rate": float(persona_resolutions[persona] / persona_counts[persona]) if persona_counts[persona] > 0 else 0.0,
                "escalation_rate": float(persona_escalations[persona] / persona_counts[persona]) if persona_counts[persona] > 0 else 0.0,
            }
            for persona in persona_counts
        },
        
        # ====================================================================
        # DETAILED EPISODE LOGS (for debugging)
        # ====================================================================
        "episode_logs": [asdict(m) for m in episode_metrics],
    }
    
    return results


# ============================================================================
# Reporting
# ============================================================================

def print_summary(results: dict[str, Any]) -> None:
    """Print human-readable summary."""
    print("\n" + "█" * 70)
    print("  EVALUATION SUMMARY")
    print("█" * 70)
    print()
    
    print("TIMING:")
    print(f"  Duration: {results['timing']['duration_seconds']:.1f}s")
    print(f"  Rate: {results['timing']['episodes_per_minute']:.1f} episodes/min")
    print()
    
    print("PRIMARY METRIC - RESOLUTION RATE:")
    print(f"  🎯 Resolution: {results['resolution_rate']*100:.1f}%")
    print()
    
    print("ALL TERMINAL OUTCOMES:")
    print(f"  ✓ Success (resolved):  {results['terminal_counts']['success']:3d} ({results['resolution_rate']*100:5.1f}%)")
    print(f"  ↑ Escalation:         {results['terminal_counts']['escalation']:3d} ({results['escalation_rate']*100:5.1f}%)")
    print(f"  ✗ Dropout:            {results['terminal_counts']['dropout']:3d} ({results['dropout_rate']*100:5.1f}%)")
    print(f"  ⏱ Timeout:            {results['terminal_counts']['timeout']:3d} ({results['timeout_rate']*100:5.1f}%)")
    print()
    
    print("REWARD DISTRIBUTION:")
    print(f"  Mean:   {results['reward']['mean']:+.3f}")
    print(f"  Std:    {results['reward']['std']:.3f}")
    print(f"  Min:    {results['reward']['min']:+.3f}")
    print(f"  Max:    {results['reward']['max']:+.3f}")
    print(f"  Median: {results['reward']['median']:+.3f}")
    print()
    
    print("TURNS TO RESOLUTION:")
    print(f"  Mean:   {results['turns_analysis']['mean_turns_to_resolution']:.1f} turns")
    print(f"  Range:  {results['turns_analysis']['min_turns']}-{results['turns_analysis']['max_turns']} turns")
    print()
    
    print("CUSTOMER SATISFACTION (PROXY):")
    print(f"  Mean:   {results['satisfaction']['mean_satisfaction_proxy']:.3f} / 1.0")
    print()
    
    print("BY TIER:")
    for tier, metrics in results['by_tier'].items():
        if metrics['count'] > 0:
            print(f"  {tier:12s}: {metrics['count']:3d} episodes | "
                  f"Resolution: {metrics['resolution_rate']*100:5.1f}% | "
                  f"Escalation: {metrics['escalation_rate']*100:5.1f}%")
    print()
    
    print("BY PERSONA:")
    for persona, metrics in results['by_persona'].items():
        if metrics['count'] > 0:
            print(f"  {persona:25s}: {metrics['count']:3d} episodes | "
                  f"Resolution: {metrics['resolution_rate']*100:5.1f}% | "
                  f"Escalation: {metrics['escalation_rate']*100:5.1f}%")
    print()
    print("█" * 70 + "\n")


# ============================================================================
# Visualization & Analysis
# ============================================================================

def create_visualizations(results: dict[str, Any], output_dir: str = ".") -> None:
    """
    Create comprehensive visualization plots from evaluation results.
    Saves plots to output_dir as PNG files.
    """
    if not PLOTTING_AVAILABLE:
        print("⚠ Skipping visualizations because matplotlib/seaborn could not be imported.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    episode_logs = results['episode_logs']
    df = pd.DataFrame(episode_logs)
    
    print("\n📊 Generating visualizations...")
    
    # ========================================================================
    # 1. Terminal Outcomes Breakdown (Pie Chart)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    sizes = [
        results['terminal_counts']['success'],
        results['terminal_counts']['escalation'],
        results['terminal_counts']['dropout'],
        results['terminal_counts']['timeout'],
    ]
    labels = [
        f"✓ Success\n({results['resolution_rate']*100:.1f}%)",
        f"↑ Escalation\n({results['escalation_rate']*100:.1f}%)",
        f"✗ Dropout\n({results['dropout_rate']*100:.1f}%)",
        f"⏱ Timeout\n({results['timeout_rate']*100:.1f}%)",
    ]
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.0f',
        startangle=90, textprops={'fontsize': 11, 'weight': 'bold'}
    )
    ax.set_title('Terminal Outcomes Distribution\n(Phase 13 LUMO Evaluation)', fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'terminal_outcomes_pie.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ terminal_outcomes_pie.png")
    
    # ========================================================================
    # 2. Resolution Rate by Tier (Bar Chart)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    tiers = list(results['by_tier'].keys())
    resolution_rates = [results['by_tier'][t]['resolution_rate']*100 for t in tiers]
    escalation_rates = [results['by_tier'][t]['escalation_rate']*100 for t in tiers]
    
    x = np.arange(len(tiers))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, resolution_rates, width, label='Resolution', color='#2ecc71')
    bars2 = ax.bar(x + width/2, escalation_rates, width, label='Escalation', color='#f39c12')
    
    ax.set_ylabel('Rate (%)', fontsize=12, weight='bold')
    ax.set_title('Performance by Tier\n(Resolution vs Escalation)', fontsize=14, weight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_tier.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ performance_by_tier.png")
    
    # ========================================================================
    # 3. Resolution Rate by Persona (Bar Chart)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    personas = list(results['by_persona'].keys())
    persona_resolution_rates = [results['by_persona'][p]['resolution_rate']*100 for p in personas]
    persona_escalation_rates = [results['by_persona'][p]['escalation_rate']*100 for p in personas]
    
    x = np.arange(len(personas))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, persona_resolution_rates, width, label='Resolution', color='#3498db')
    bars2 = ax.bar(x + width/2, persona_escalation_rates, width, label='Escalation', color='#e67e22')
    
    ax.set_ylabel('Rate (%)', fontsize=12, weight='bold')
    ax.set_title('Performance by Customer Persona\n(Resolution vs Escalation)', fontsize=14, weight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('_', '\n') for p in personas], fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_persona.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ performance_by_persona.png")
    
    # ========================================================================
    # 4. Reward Distribution (Histogram + KDE)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(11, 6))
    rewards = df['reward'].values
    
    ax.hist(rewards, bins=30, alpha=0.7, color='#3498db', edgecolor='black', density=True)
    
    # Add KDE
    from scipy import stats
    kde = stats.gaussian_kde(rewards)
    x_range = np.linspace(rewards.min(), rewards.max(), 200)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    ax.axvline(np.mean(rewards), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.3f}')
    ax.axvline(np.median(rewards), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.3f}')
    
    ax.set_xlabel('Cumulative Reward', fontsize=12, weight='bold')
    ax.set_ylabel('Density', fontsize=12, weight='bold')
    ax.set_title('Reward Distribution Across All Episodes', fontsize=14, weight='bold', pad=15)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reward_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ reward_distribution.png")
    
    # ========================================================================
    # 5. Turns to Resolution (Histogram)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    turns_to_resolution = df[df['terminal_type'] == 'success']['turns'].values
    
    if len(turns_to_resolution) > 0:
        ax.hist(turns_to_resolution, bins=range(1, int(turns_to_resolution.max())+2), 
               alpha=0.7, color='#2ecc71', edgecolor='black', align='left')
        
        ax.axvline(np.mean(turns_to_resolution), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(turns_to_resolution):.1f}')
        
        ax.set_xlabel('Number of Turns', fontsize=12, weight='bold')
        ax.set_ylabel('Frequency', fontsize=12, weight='bold')
        ax.set_title(f'Turns to Resolution (n={len(turns_to_resolution)} successes)', 
                    fontsize=14, weight='bold', pad=15)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'turns_to_resolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ turns_to_resolution.png")
    
    # ========================================================================
    # 6. Reward Over Episode (Trend Line)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    episodes = df['episode_id'].values
    rewards = df['reward'].values
    
    # Color by terminal type
    colors_map = {
        'success': '#2ecc71',
        'escalation': '#f39c12',
        'dropout': '#e74c3c',
        'timeout': '#95a5a6',
    }
    colors = [colors_map.get(t, 'gray') for t in df['terminal_type']]
    
    ax.scatter(episodes, rewards, c=colors, alpha=0.6, s=50, edgecolors='none')
    
    # Add rolling average
    window = max(1, len(rewards) // 10)
    rolling_mean = pd.Series(rewards).rolling(window=window, center=True).mean()
    ax.plot(episodes, rolling_mean, color='red', linewidth=2.5, label=f'Rolling Avg (window={window})')
    
    ax.set_xlabel('Episode', fontsize=12, weight='bold')
    ax.set_ylabel('Cumulative Reward', fontsize=12, weight='bold')
    ax.set_title('Reward Trajectory Over Episodes', fontsize=14, weight='bold', pad=15)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_map[k], label=k) for k in colors_map]
    ax.legend(handles=legend_elements + [plt.Line2D([0], [0], color='red', linewidth=2.5, label=f'Rolling Avg')], 
             fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'episode_rewards_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ episode_rewards_trajectory.png")
    
    # ========================================================================
    # 7. Satisfaction Proxy Distribution
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    satisfactions = df['satisfaction_proxy'].values
    
    ax.hist(satisfactions, bins=20, alpha=0.7, color='#9b59b6', edgecolor='black')
    ax.axvline(np.mean(satisfactions), color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {np.mean(satisfactions):.3f}')
    
    ax.set_xlabel('Satisfaction Proxy Score', fontsize=12, weight='bold')
    ax.set_ylabel('Frequency', fontsize=12, weight='bold')
    ax.set_title('Customer Satisfaction (Proxy) Distribution\n(0.0-1.0 Scale)', 
                fontsize=14, weight='bold', pad=15)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'satisfaction_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ satisfaction_distribution.png")
    
    # ========================================================================
    # 8. Terminal Type by Tier (Heatmap-like Stacked Bar)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    tier_data = {}
    for tier in results['by_tier']:
        tier_episodes = df[df['tier'] == tier]
        tier_data[tier] = {
            'success': len(tier_episodes[tier_episodes['terminal_type'] == 'success']),
            'escalation': len(tier_episodes[tier_episodes['terminal_type'] == 'escalation']),
            'dropout': len(tier_episodes[tier_episodes['terminal_type'] == 'dropout']),
            'timeout': len(tier_episodes[tier_episodes['terminal_type'] == 'timeout']),
        }
    
    tiers = list(tier_data.keys())
    success_counts = [tier_data[t]['success'] for t in tiers]
    escalation_counts = [tier_data[t]['escalation'] for t in tiers]
    dropout_counts = [tier_data[t]['dropout'] for t in tiers]
    timeout_counts = [tier_data[t]['timeout'] for t in tiers]
    
    x = np.arange(len(tiers))
    width = 0.6
    
    ax.bar(x, success_counts, width, label='Success', color='#2ecc71')
    ax.bar(x, escalation_counts, width, bottom=success_counts, label='Escalation', color='#f39c12')
    ax.bar(x, dropout_counts, width, 
          bottom=[s+e for s,e in zip(success_counts, escalation_counts)], 
          label='Dropout', color='#e74c3c')
    ax.bar(x, timeout_counts, width,
          bottom=[s+e+d for s,e,d in zip(success_counts, escalation_counts, dropout_counts)],
          label='Timeout', color='#95a5a6')
    
    ax.set_ylabel('Count', fontsize=12, weight='bold')
    ax.set_title('Terminal Outcomes by Tier (Stacked)', fontsize=14, weight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'terminal_outcomes_by_tier_stacked.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ terminal_outcomes_by_tier_stacked.png")
    
    print(f"✓ All visualizations saved to: {output_dir}\n")


def generate_statistical_analysis(results: dict[str, Any]) -> str:
    """
    Generate statistical analysis summary as text.
    """
    episode_logs = results['episode_logs']
    df = pd.DataFrame(episode_logs)
    
    analysis = []
    analysis.append("\n" + "="*70)
    analysis.append("STATISTICAL ANALYSIS")
    analysis.append("="*70 + "\n")
    
    # ========================================================================
    # Overall Performance
    # ========================================================================
    analysis.append("1. OVERALL PERFORMANCE")
    analysis.append("-" * 70)
    analysis.append(f"  Total Episodes: {results['configuration']['n_episodes']}")
    analysis.append(f"  Duration: {results['timing']['duration_seconds']:.1f}s ({results['timing']['episodes_per_minute']:.1f} ep/min)")
    analysis.append(f"  Primary Metric (Resolution): {results['resolution_rate']*100:.2f}%")
    analysis.append(f"  Expected Resolution for Random Policy: ~25%")
    analysis.append(f"  Improvement Factor: {(results['resolution_rate']/0.25):.2f}x")
    analysis.append("")
    
    # ========================================================================
    # Reward Analysis
    # ========================================================================
    analysis.append("2. REWARD ANALYSIS")
    analysis.append("-" * 70)
    analysis.append(f"  Mean Reward: {results['reward']['mean']:+.4f}")
    analysis.append(f"  Std Dev: {results['reward']['std']:.4f}")
    analysis.append(f"  Range: [{results['reward']['min']:+.4f}, {results['reward']['max']:+.4f}]")
    analysis.append(f"  Coefficient of Variation: {(results['reward']['std']/abs(results['reward']['mean'])):.2f}" 
                   if results['reward']['mean'] != 0 else "  CoV: undefined (mean=0)")
    analysis.append("")
    
    # ========================================================================
    # Efficiency Analysis
    # ========================================================================
    analysis.append("3. EFFICIENCY ANALYSIS")
    analysis.append("-" * 70)
    turn_stats = results['turns_analysis']
    analysis.append(f"  Mean Turns to Resolution: {turn_stats['mean_turns_to_resolution']:.1f}")
    analysis.append(f"  Std Dev: {turn_stats['std_turns_to_resolution']:.1f}")
    analysis.append(f"  Range: {turn_stats['min_turns']}-{turn_stats['max_turns']} turns")
    
    if turn_stats['mean_turns_to_resolution'] > 0:
        efficiency = 1.0 / turn_stats['mean_turns_to_resolution']
        analysis.append(f"  Efficiency (1/avg_turns): {efficiency:.3f}")
    analysis.append("")
    
    # ========================================================================
    # Satisfaction Analysis
    # ========================================================================
    analysis.append("4. CUSTOMER SATISFACTION (PROXY)")
    analysis.append("-" * 70)
    analysis.append(f"  Mean Satisfaction: {results['satisfaction']['mean_satisfaction_proxy']:.3f}/1.0")
    analysis.append(f"  Std Dev: {results['satisfaction']['std_satisfaction_proxy']:.3f}")
    analysis.append(f"  Interpretation: {['Very Poor', 'Poor', 'Fair', 'Good', 'Very Good'][min(4, int(results['satisfaction']['mean_satisfaction_proxy']*5))]}")
    analysis.append("")
    
    # ========================================================================
    # Tier Analysis
    # ========================================================================
    analysis.append("5. PERFORMANCE BY TIER")
    analysis.append("-" * 70)
    for tier, metrics in results['by_tier'].items():
        if metrics['count'] > 0:
            analysis.append(f"  {tier:12s}: n={metrics['count']:3d} | "
                          f"Res={metrics['resolution_rate']*100:5.1f}% | "
                          f"Esc={metrics['escalation_rate']*100:5.1f}%")
    analysis.append("")
    
    # ========================================================================
    # Persona Analysis
    # ========================================================================
    analysis.append("6. PERFORMANCE BY PERSONA")
    analysis.append("-" * 70)
    for persona, metrics in results['by_persona'].items():
        if metrics['count'] > 0:
            analysis.append(f"  {persona:25s}: n={metrics['count']:3d} | "
                          f"Res={metrics['resolution_rate']*100:5.1f}% | "
                          f"Esc={metrics['escalation_rate']*100:5.1f}%")
    analysis.append("")
    
    # ========================================================================
    # Statistical Significance
    # ========================================================================
    analysis.append("7. KEY OBSERVATIONS")
    analysis.append("-" * 70)
    
    # Success vs other outcomes
    success_ratio = results['terminal_counts']['success'] / results['configuration']['n_episodes']
    escalation_ratio = results['terminal_counts']['escalation'] / results['configuration']['n_episodes']
    dropout_ratio = results['terminal_counts']['dropout'] / results['configuration']['n_episodes']
    
    if success_ratio == 0:
        analysis.append(f"  ⚠  Model failed to resolve any cases (0% resolution, {escalation_ratio*100:.1f}% escalation)")
    elif escalation_ratio < success_ratio * 0.5:
        analysis.append(f"  ✓ Model heavily favors resolution (escalation {escalation_ratio/success_ratio:.1%} of success rate)")
    elif escalation_ratio > success_ratio:
        analysis.append(f"  ⚠  Model escalates more than it resolves (escalation {escalation_ratio/success_ratio:.1%}x success)")
    else:
        analysis.append(f"  ~ Balanced escalation strategy")
    
    if dropout_ratio < 0.05:
        analysis.append(f"  ✓ Very low dropout rate ({dropout_ratio*100:.1f}%)")
    elif dropout_ratio > 0.15:
        analysis.append(f"  ⚠  High dropout rate ({dropout_ratio*100:.1f}%) - customers leaving")
    else:
        analysis.append(f"  ~ Moderate dropout rate ({dropout_ratio*100:.1f}%)")
    
    if results['satisfaction']['mean_satisfaction_proxy'] > 0.7:
        analysis.append(f"  ✓ Strong satisfaction proxy ({results['satisfaction']['mean_satisfaction_proxy']:.3f})")
    elif results['satisfaction']['mean_satisfaction_proxy'] > 0.5:
        analysis.append(f"  ~ Moderate satisfaction proxy ({results['satisfaction']['mean_satisfaction_proxy']:.3f})")
    else:
        analysis.append(f"  ⚠  Low satisfaction proxy ({results['satisfaction']['mean_satisfaction_proxy']:.3f})")
    
    analysis.append("")
    analysis.append("="*70 + "\n")
    
    return "\n".join(analysis)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 13 Full LUMO-Based Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python phase13_lumo_evaluation.py --episodes 25
  python phase13_lumo_evaluation.py --episodes 100 --output phase13_eval_100.json
  python phase13_lumo_evaluation.py --episodes 250 --output phase13_eval_250.json
  python phase13_lumo_evaluation.py --episodes 100 --plots ./results_plots
        """
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to run (default: 100)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="JSON file to save results (default: phase13_evaluation_<timestamp>.json)"
    )
    parser.add_argument(
        "--plots",
        type=str,
        default=None,
        help="Directory to save plots (default: same as output directory)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RL Out/models/best_model.zip",
        help="Path to model (default: RL Out/models/best_model.zip)"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_phase13_lumo(
        model_path=args.model,
        artifacts_root="Simulation_4/artifacts",
        n_episodes=args.episodes,
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"phase13_evaluation_{args.episodes}ep_{timestamp}.json"
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {args.output}\n")
    
    # Generate and save analysis
    analysis_text = generate_statistical_analysis(results)
    print(analysis_text)
    
    analysis_file = str(args.output).replace('.json', '_analysis.txt')
    with open(analysis_file, "w") as f:
        f.write(analysis_text)
    print(f"✓ Analysis saved to: {analysis_file}\n")
    
    # Create visualizations
    if args.plots is None:
        args.plots = str(Path(args.output).parent)
    
    create_visualizations(results, args.plots)
    if PLOTTING_AVAILABLE:
        print(f"📊 All outputs generated successfully!\n")
    else:
        print(f"✓ Evaluation completed without plots.\n")
