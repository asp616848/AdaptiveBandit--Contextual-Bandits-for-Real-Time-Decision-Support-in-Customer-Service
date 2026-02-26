"""
Training Pipeline for Customer Support Bandit.

Runs experiments comparing:
1. Rule-based baseline
2. LinUCB contextual bandit
3. Thompson Sampling contextual bandit
4. DQN (full MDP)

Evaluates using business-aligned metrics (cost saved, CSAT, profit).
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import TRAINING_CONFIG, TIER_NAMES, TIER_CONFIG, DQN_CONFIG
from data.feature_engineer import FEATURE_DIM
from environment.customer_env import CustomerSupportEnv
from agents.base_agent import BaseAgent
from agents.rule_based import RuleBasedAgent
from agents.linucb import LinUCBAgent
from agents.thompson_sampling import ThompsonSamplingAgent
from agents.dqn_agent import DQNAgent


def train_bandit(agent: BaseAgent,
                 env: CustomerSupportEnv,
                 n_episodes: int = 5000,
                 eval_every: int = 100,
                 verbose: bool = True) -> Dict:
    """
    Train a contextual bandit agent on the customer support environment.

    Parameters
    ----------
    agent : BaseAgent
        Bandit agent (LinUCB, Thompson Sampling, or Rule-based).
    env : CustomerSupportEnv
        Environment in bandit mode.
    n_episodes : int
        Number of training episodes (conversations).
    eval_every : int
        Evaluate and log every N episodes.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Training history with rewards, actions, and evaluation metrics.
    """
    assert env.mode == "bandit", "Environment must be in bandit mode"

    history = {
        'rewards': [],
        'actions': [],
        'outcomes': [],
        'cumulative_reward': [],
        'eval_points': [],
        'eval_rewards': [],
        'tiers': [],
    }

    cumulative = 0.0
    start = time.time()

    for ep in range(n_episodes):
        obs = env.reset()
        action = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        agent.update(obs, action, reward)

        history['rewards'].append(reward)
        history['actions'].append(action)
        history['outcomes'].append(info.get('outcome', ''))
        history['tiers'].append(info.get('tier', ''))
        cumulative += reward
        history['cumulative_reward'].append(cumulative)

        if (ep + 1) % eval_every == 0:
            avg_reward = np.mean(history['rewards'][-eval_every:])
            history['eval_points'].append(ep + 1)
            history['eval_rewards'].append(avg_reward)

            if verbose:
                elapsed = time.time() - start
                policy = agent.get_policy_info()
                actions_dist = policy['action_distribution']
                print(f"  Episode {ep+1:5d} | "
                      f"Avg Reward: {avg_reward:+.3f} | "
                      f"Bot/Human: {actions_dist[0]:.2f}/{actions_dist[1]:.2f} | "
                      f"Time: {elapsed:.1f}s")

    total_time = time.time() - start
    history['total_time'] = total_time
    history['final_policy'] = agent.get_policy_info()

    if verbose:
        print(f"\n  Training complete in {total_time:.1f}s")
        print(f"  Final avg reward: {np.mean(history['rewards'][-500:]):+.3f}")

    return history


def train_dqn(agent: DQNAgent,
              env: CustomerSupportEnv,
              n_episodes: int = 5000,
              eval_every: int = 100,
              verbose: bool = True) -> Dict:
    """
    Train a DQN agent on the multi-turn MDP environment.

    Parameters
    ----------
    agent : DQNAgent
        DQN agent.
    env : CustomerSupportEnv
        Environment in MDP mode.
    n_episodes : int
        Number of training episodes.
    eval_every : int
        Evaluate and log every N episodes.

    Returns
    -------
    dict
        Training history.
    """
    assert env.mode == "mdp", "Environment must be in MDP mode"

    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'losses': [],
        'eval_points': [],
        'eval_rewards': [],
        'epsilon_history': [],
    }

    start = time.time()

    for ep in range(n_episodes):
        obs = env.reset()
        episode_reward = 0.0
        episode_length = 0

        while not env.done:
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)

            loss = agent.update(obs, action, reward, next_obs, done)
            if loss is not None:
                history['losses'].append(loss)

            obs = next_obs
            episode_reward += reward
            episode_length += 1

        history['episode_rewards'].append(episode_reward)
        history['episode_lengths'].append(episode_length)
        history['epsilon_history'].append(agent.epsilon)

        if (ep + 1) % eval_every == 0:
            avg_reward = np.mean(history['episode_rewards'][-eval_every:])
            avg_length = np.mean(history['episode_lengths'][-eval_every:])
            history['eval_points'].append(ep + 1)
            history['eval_rewards'].append(avg_reward)

            if verbose:
                avg_loss = np.mean(history['losses'][-100:]) if history['losses'] else 0
                print(f"  Episode {ep+1:5d} | "
                      f"Avg Reward: {avg_reward:+.3f} | "
                      f"Avg Length: {avg_length:.1f} | "
                      f"Epsilon: {agent.epsilon:.3f} | "
                      f"Loss: {avg_loss:.4f}")

    history['total_time'] = time.time() - start
    history['final_policy'] = agent.get_policy_info()

    return history


def run_experiment(conversations: Optional[List[Dict]] = None,
                   n_episodes: int = 5000,
                   seed: int = 42,
                   verbose: bool = True) -> Dict:
    """
    Run the full experiment comparing all agents.

    Parameters
    ----------
    conversations : list of dict, optional
        Real conversation data. Uses synthetic if None.
    n_episodes : int
        Training episodes per agent.
    seed : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Results for all agents.
    """
    np.random.seed(seed)
    results = {}

    # ─── Phase I: Contextual Bandit Comparison ───
    if verbose:
        print("=" * 60)
        print("PHASE I: Contextual Bandit Comparison")
        print("=" * 60)

    # Create bandit environment
    bandit_env = CustomerSupportEnv(
        mode="bandit", conversations=conversations, seed=seed
    )

    # 1. Rule-based baseline
    if verbose:
        print("\n[1/4] Rule-Based Baseline")
    rule_agent = RuleBasedAgent(feature_dim=FEATURE_DIM)
    results['rule_based'] = train_bandit(
        rule_agent, bandit_env, n_episodes, verbose=verbose
    )

    # 2. LinUCB
    if verbose:
        print("\n[2/4] LinUCB (alpha=1.0)")
    bandit_env_linucb = CustomerSupportEnv(
        mode="bandit", conversations=conversations, seed=seed
    )
    linucb_agent = LinUCBAgent(feature_dim=FEATURE_DIM, alpha=1.0)
    results['linucb'] = train_bandit(
        linucb_agent, bandit_env_linucb, n_episodes, verbose=verbose
    )

    # 3. Thompson Sampling
    if verbose:
        print("\n[3/4] Thompson Sampling")
    bandit_env_ts = CustomerSupportEnv(
        mode="bandit", conversations=conversations, seed=seed
    )
    ts_agent = ThompsonSamplingAgent(feature_dim=FEATURE_DIM)
    results['thompson_sampling'] = train_bandit(
        ts_agent, bandit_env_ts, n_episodes, verbose=verbose
    )

    # ─── Phase II: Full RL (DQN) ───
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE II: Full RL (DQN)")
        print("=" * 60)

    # 4. DQN
    if verbose:
        print("\n[4/4] DQN Agent")
    mdp_env = CustomerSupportEnv(
        mode="mdp", conversations=conversations, seed=seed
    )
    dqn_agent = DQNAgent(feature_dim=FEATURE_DIM)
    results['dqn'] = train_dqn(
        dqn_agent, mdp_env, n_episodes, verbose=verbose
    )

    # Store agents for later analysis
    results['agents'] = {
        'rule_based': rule_agent,
        'linucb': linucb_agent,
        'thompson_sampling': ts_agent,
        'dqn': dqn_agent,
    }

    return results


def evaluate_on_tiers(agent: BaseAgent,
                      env: CustomerSupportEnv,
                      n_eval: int = 500) -> Dict:
    """
    Evaluate agent performance broken down by tier.

    Returns per-tier metrics: avg reward, escalation rate,
    correct routing rate.
    """
    tier_results = {tier: {'rewards': [], 'actions': [], 'outcomes': []}
                    for tier in TIER_NAMES}

    for _ in range(n_eval):
        obs = env.reset()
        action = agent.select_action(obs)
        _, reward, _, info = env.step(action)

        tier = info.get('tier', 'Free')
        tier_results[tier]['rewards'].append(reward)
        tier_results[tier]['actions'].append(action)
        tier_results[tier]['outcomes'].append(info.get('outcome', ''))

    summary = {}
    for tier, data in tier_results.items():
        if len(data['rewards']) == 0:
            continue
        actions = np.array(data['actions'])
        rewards = np.array(data['rewards'])
        outcomes = data['outcomes']

        correct = sum(1 for o in outcomes if o.startswith('correct_'))
        total = len(outcomes)

        summary[tier] = {
            'n_conversations': total,
            'avg_reward': np.mean(rewards),
            'escalation_rate': np.mean(actions == 1),
            'correct_routing_rate': correct / max(total, 1),
            'total_reward': np.sum(rewards),
        }

    return summary
