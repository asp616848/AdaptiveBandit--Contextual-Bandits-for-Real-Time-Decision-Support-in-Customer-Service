"""
Data Pipeline — Converts raw datasets into the unified conversation format
expected by CustomerSupportEnv.

Processes:
1. Twitter CS → conversation threads with escalation labels
2. OpenAssistant → conversation trees with quality ratings
3. Merges at feature level into unified training conversations

Output format per conversation:
{
    'texts': List[str],           # message texts in order
    'tier': str,                  # simulated SaaS tier
    'escalation_needed': int,     # 0/1 proxy label
    'num_turns': int,
    'avg_quality': float,         # from OA or estimated
    'source': str,                # 'twitter' | 'openassistant' | 'synthetic'
    'conversation_id': str|int,
}
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import DATA_PATHS, TIER_NAMES
from data.feature_engineer import (
    assign_simulated_tier, compute_sentiment,
    compute_escalation_features,
)


# ═══════════════════════════════════════════════════════════
# Twitter → Conversations
# ═══════════════════════════════════════════════════════════

def twitter_to_conversations(sample_n: Optional[int] = 5000,
                              seed: int = 42) -> List[Dict]:
    """
    Load Twitter CS data and convert to conversation format.

    Parameters
    ----------
    sample_n : int, optional
        Number of threads to include (None = all threads, slow).
    seed : int
        Random seed for sampling and tier assignment.

    Returns
    -------
    list of dict
        Conversations in the unified format.
    """
    from data.twitter_loader import (
        load_twitter_data,
        reconstruct_threads,
        add_escalation_labels,
    )

    rng = np.random.RandomState(seed)

    # Load raw data — sample for speed on large dataset
    frac = min(1.0, (sample_n or 50000) * 5 / 2_800_000)
    df = load_twitter_data(sample_frac=frac, random_state=seed)

    # Reconstruct threads
    threads = reconstruct_threads(df)

    # Add escalation labels
    threads = add_escalation_labels(threads)

    # Sample if requested
    if sample_n and len(threads) > sample_n:
        threads = threads.sample(n=sample_n, random_state=seed).reset_index(drop=True)

    # Convert to conversation list
    conversations = []
    for _, row in threads.iterrows():
        texts = row.get('tweets', [])
        if not isinstance(texts, list) or len(texts) == 0:
            continue

        conv = {
            'texts': texts,
            'tier': assign_simulated_tier(rng),
            'escalation_needed': int(row.get('escalation_needed', 0)),
            'num_turns': int(row.get('num_turns', len(texts))),
            'avg_quality': 0.7 if row.get('has_brand_reply', True) else 0.3,
            'source': 'twitter',
            'conversation_id': f"tw_{row.get('thread_id', '')}",
        }
        conversations.append(conv)

    print(f"Converted {len(conversations):,} Twitter conversations")
    return conversations


# ═══════════════════════════════════════════════════════════
# OpenAssistant → Conversations
# ═══════════════════════════════════════════════════════════

def openassistant_to_conversations(sample_n: Optional[int] = 3000,
                                     seed: int = 42) -> List[Dict]:
    """
    Load OpenAssistant data and convert to conversation format.

    Uses quality ratings to derive escalation proxy: low quality → escalation needed.
    """
    from data.openassistant_loader import (
        load_openassistant_data,
        build_conversation_trees,
    )

    rng = np.random.RandomState(seed)

    # Load data
    df = load_openassistant_data(use_parquet=True, sample_frac=0.3, random_state=seed)

    # Build conversation trees
    trees = build_conversation_trees(df)

    if sample_n and len(trees) > sample_n:
        trees = trees.sample(n=sample_n, random_state=seed).reset_index(drop=True)

    # Quality threshold for escalation proxy
    quality_threshold = trees['avg_quality'].quantile(0.25)

    conversations = []
    for _, row in trees.iterrows():
        texts = row.get('texts', [])
        if not isinstance(texts, list) or len(texts) == 0:
            continue

        quality = row.get('avg_quality', np.nan)

        # Escalation proxy: low quality → needed escalation
        if np.isnan(quality):
            esc_needed = int(rng.random() < 0.35)
        else:
            esc_needed = int(quality < quality_threshold)

        conv = {
            'texts': texts[:15],  # Truncate very long trees
            'tier': assign_simulated_tier(rng),
            'escalation_needed': esc_needed,
            'num_turns': min(int(row.get('num_turns', len(texts))), 15),
            'avg_quality': float(quality) if not np.isnan(quality) else 0.5,
            'source': 'openassistant',
            'conversation_id': f"oa_{row.get('tree_id', '')}",
        }
        conversations.append(conv)

    print(f"Converted {len(conversations):,} OpenAssistant conversations")
    return conversations


# ═══════════════════════════════════════════════════════════
# Synthetic fallback (when real data unavailable)
# ═══════════════════════════════════════════════════════════

def generate_synthetic_conversations(n: int = 2000,
                                       seed: int = 42) -> List[Dict]:
    """
    Generate synthetic conversations that mimic real support patterns.

    Uses statistical properties observed from the actual datasets:
    - Median ~4 turns, right-skewed (geometric distribution)
    - ~35% need escalation
    - Tier distribution: Free 60%, Pro 25%, Business+ 12%, Enterprise 3%
    """
    rng = np.random.RandomState(seed)

    # Templates covering range of support issues
    easy_openers = [
        "Hi, I need help resetting my password.",
        "Where can I find the billing settings?",
        "How do I add a new team member to my workspace?",
        "Can you tell me about the keyboard shortcuts?",
        "I'd like to change my notification preferences.",
        "What's the difference between channels and DMs?",
        "How do I set my status message?",
        "Can I export my chat history?",
        "How do I create a new channel for our project?",
        "Is there a way to schedule messages?",
    ]

    hard_openers = [
        "Your service has been down for 3 hours and we're losing money!",
        "I've been trying to get help for days and nobody responds!",
        "The API keeps returning 500 errors. Our production is completely broken.",
        "I want to speak to a manager immediately. This is unacceptable.",
        "We're considering canceling unless this integration issue is fixed TODAY.",
        "I was charged twice on my credit card. I need a refund right now!",
        "Our enterprise team can't access any files since the last update.",
        "The SSO integration broke and 500 employees are locked out.",
        "This is the third time I've reported this bug. Still not fixed.",
        "We need to escalate this immediately — data privacy issue.",
    ]

    easy_followups = [
        "Thanks, that worked!",
        "Got it, appreciate the help.",
        "Perfect, exactly what I needed.",
        "Makes sense, thanks!",
        "Great, I'll try that now.",
    ]

    hard_followups = [
        "Still not working. This is really frustrating.",
        "That didn't help at all. I need a real solution.",
        "Can someone who actually knows what they're doing help me?",
        "I've already tried that. I said this earlier.",
        "This is taking too long. I'm running out of patience.",
        "Our CEO is asking about this. It's becoming critical.",
        "If this isn't resolved today I'm filing a formal complaint.",
    ]

    agent_responses = [
        "I understand your concern. Let me look into this right away.",
        "I'm sorry for the inconvenience. Here's what you can try:",
        "Thank you for your patience. I've escalated this to our engineering team.",
        "I can see the issue. Let me walk you through the fix.",
        "I appreciate you bringing this to our attention. Here's the solution:",
    ]

    conversations = []
    for i in range(n):
        is_hard = rng.random() < 0.35
        tier = assign_simulated_tier(rng)

        # Number of turns (geometric, matching real data distribution)
        n_turns = min(rng.geometric(0.25) + 1, 15)

        if is_hard:
            opener = rng.choice(hard_openers)
            followups = hard_followups
        else:
            opener = rng.choice(easy_openers)
            followups = easy_followups

        texts = [opener]
        for j in range(n_turns - 1):
            if j % 2 == 0:
                texts.append(rng.choice(agent_responses))
            else:
                texts.append(rng.choice(followups))

        # Quality: correlated with difficulty and tier
        if is_hard:
            quality = rng.beta(2, 5)  # Skewed low
        else:
            quality = rng.beta(5, 2)  # Skewed high

        conversations.append({
            'texts': texts,
            'tier': tier,
            'escalation_needed': int(is_hard),
            'num_turns': len(texts),
            'avg_quality': float(quality),
            'source': 'synthetic',
            'conversation_id': f"syn_{i}",
        })

    # Print distribution stats
    esc_rate = np.mean([c['escalation_needed'] for c in conversations])
    avg_turns = np.mean([c['num_turns'] for c in conversations])
    tier_dist = pd.Series([c['tier'] for c in conversations]).value_counts(normalize=True)

    print(f"Generated {n:,} synthetic conversations")
    print(f"  Escalation rate: {esc_rate:.1%}")
    print(f"  Avg turns: {avg_turns:.1f}")
    print(f"  Tier distribution:")
    for tier in TIER_NAMES:
        pct = tier_dist.get(tier, 0)
        print(f"    {tier}: {pct:.1%}")

    return conversations


# ═══════════════════════════════════════════════════════════
# Unified Pipeline
# ═══════════════════════════════════════════════════════════

def build_unified_dataset(use_twitter: bool = True,
                           use_openassistant: bool = True,
                           use_synthetic: bool = True,
                           twitter_n: Optional[int] = 3000,
                           oa_n: Optional[int] = 2000,
                           synthetic_n: int = 2000,
                           seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Build unified training and evaluation datasets.

    Combines conversations from all available sources and splits
    into train (80%) and eval (20%).

    Parameters
    ----------
    use_twitter : bool
        Include Twitter CS conversations.
    use_openassistant : bool
        Include OpenAssistant conversations.
    use_synthetic : bool
        Include synthetic conversations (fallback/supplement).
    twitter_n : int, optional
        Max Twitter conversations to include.
    oa_n : int, optional
        Max OpenAssistant conversations to include.
    synthetic_n : int
        Number of synthetic conversations.
    seed : int
        Random seed.

    Returns
    -------
    train_convs : list of dict
        Training conversations (~80%).
    eval_convs : list of dict
        Evaluation conversations (~20%).
    """
    rng = np.random.RandomState(seed)
    all_conversations = []

    # Twitter
    if use_twitter:
        try:
            twitter_path = DATA_PATHS.get("twitter", "")
            if Path(twitter_path).exists():
                tw_convs = twitter_to_conversations(sample_n=twitter_n, seed=seed)
                all_conversations.extend(tw_convs)
            else:
                print(f"Twitter data not found at {twitter_path}, skipping.")
        except Exception as e:
            print(f"Error loading Twitter data: {e}")

    # OpenAssistant
    if use_openassistant:
        try:
            oa_path = DATA_PATHS.get("openassistant_train", "")
            oa_csv = DATA_PATHS.get("openassistant_train_csv", "")
            if Path(oa_path).exists() or Path(oa_csv).exists():
                oa_convs = openassistant_to_conversations(sample_n=oa_n, seed=seed)
                all_conversations.extend(oa_convs)
            else:
                print(f"OpenAssistant data not found, skipping.")
        except Exception as e:
            print(f"Error loading OpenAssistant data: {e}")

    # Synthetic (always available)
    if use_synthetic or len(all_conversations) == 0:
        syn_convs = generate_synthetic_conversations(n=synthetic_n, seed=seed)
        all_conversations.extend(syn_convs)

    # Shuffle
    rng.shuffle(all_conversations)

    # Split 80/20
    split_idx = int(len(all_conversations) * 0.8)
    train_convs = all_conversations[:split_idx]
    eval_convs = all_conversations[split_idx:]

    print(f"\nUnified Dataset Summary:")
    print(f"  Total: {len(all_conversations):,} conversations")
    print(f"  Train: {len(train_convs):,} | Eval: {len(eval_convs):,}")

    # Source breakdown
    sources = pd.Series([c['source'] for c in all_conversations]).value_counts()
    for src, cnt in sources.items():
        print(f"  {src}: {cnt:,} ({cnt/len(all_conversations):.1%})")

    return train_convs, eval_convs
