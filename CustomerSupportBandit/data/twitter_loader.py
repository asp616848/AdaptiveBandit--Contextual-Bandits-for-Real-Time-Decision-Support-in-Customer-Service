"""
Twitter Customer Support dataset loader.

Loads ~2.8M tweets, reconstructs conversation threads, and derives
escalation signals (unanswered tweets, escalation phrases, sentiment).
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_PATHS, ESCALATION_PHRASES


def load_twitter_data(sample_frac: Optional[float] = None,
                      random_state: int = 42) -> pd.DataFrame:
    """
    Load the Twitter Customer Support dataset.

    Parameters
    ----------
    sample_frac : float, optional
        Fraction of data to sample (for faster iteration). None = full dataset.
    random_state : int
        Random seed for sampling.

    Returns
    -------
    pd.DataFrame
        Loaded and basic-cleaned Twitter CS data.
    """
    path = DATA_PATHS["twitter"]
    print(f"Loading Twitter CS data from {path} ...")
    df = pd.read_csv(path)

    if sample_frac is not None and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
        print(f"  Sampled {len(df):,} rows ({sample_frac*100:.0f}%)")

    # Basic type fixes
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['inbound'] = df['inbound'].astype(bool) if 'inbound' in df.columns else None

    print(f"  Loaded {len(df):,} tweets | "
          f"Inbound: {df['inbound'].sum():,} | "
          f"Outbound: {(~df['inbound']).sum():,}")
    return df


def reconstruct_threads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct conversation threads by chaining reply IDs.

    Returns a DataFrame with one row per conversation thread containing:
    - thread_id: unique conversation identifier
    - tweets: list of tweet texts in order
    - num_turns: number of turns
    - brand: author_id of the brand (outbound sender)
    - has_brand_reply: whether brand responded
    - customer_texts: concatenated customer messages
    - brand_texts: concatenated brand messages
    """
    print("Reconstructing conversation threads ...")

    # Build reply chain lookup
    reply_map = {}
    for _, row in df.iterrows():
        if pd.notna(row.get('in_response_to_tweet_id')):
            child_id = row['tweet_id']
            parent_id = row['in_response_to_tweet_id']
            reply_map[child_id] = parent_id

    # Find root tweets (inbound tweets with no parent)
    roots = df[
        (df['inbound'] == True) &
        (df['in_response_to_tweet_id'].isna())
    ]['tweet_id'].values

    # Build threads from roots
    tweet_lookup = df.set_index('tweet_id')

    # Build children map
    children_map = {}
    for child_id, parent_id in reply_map.items():
        if parent_id not in children_map:
            children_map[parent_id] = []
        children_map[parent_id].append(child_id)

    threads = []
    for root_id in roots:
        thread_tweets = []
        queue = [root_id]
        while queue:
            tid = queue.pop(0)
            if tid in tweet_lookup.index:
                row = tweet_lookup.loc[tid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                thread_tweets.append({
                    'tweet_id': tid,
                    'text': str(row.get('text', '')),
                    'inbound': bool(row.get('inbound', True)),
                    'author_id': row.get('author_id', ''),
                    'created_at': row.get('created_at', None),
                })
            if tid in children_map:
                queue.extend(children_map[tid])

        if len(thread_tweets) == 0:
            continue

        # Sort by time
        thread_tweets.sort(key=lambda x: x.get('created_at') or pd.Timestamp.min)

        customer_texts = [t['text'] for t in thread_tweets if t['inbound']]
        brand_texts = [t['text'] for t in thread_tweets if not t['inbound']]

        # Identify brand
        brands = [t['author_id'] for t in thread_tweets if not t['inbound']]
        brand = brands[0] if brands else None

        threads.append({
            'thread_id': root_id,
            'tweets': [t['text'] for t in thread_tweets],
            'num_turns': len(thread_tweets),
            'brand': brand,
            'has_brand_reply': len(brand_texts) > 0,
            'customer_texts': ' '.join(customer_texts),
            'brand_texts': ' '.join(brand_texts),
            'num_customer_msgs': len(customer_texts),
            'num_brand_msgs': len(brand_texts),
        })

    threads_df = pd.DataFrame(threads)
    print(f"  Reconstructed {len(threads_df):,} threads "
          f"(avg {threads_df['num_turns'].mean():.1f} turns)")
    return threads_df


def detect_escalation_phrases(text: str) -> Tuple[bool, List[str]]:
    """Check if text contains any escalation indicator phrases."""
    if not isinstance(text, str):
        return False, []
    text_lower = text.lower()
    found = [phrase for phrase in ESCALATION_PHRASES if phrase in text_lower]
    return len(found) > 0, found


def add_escalation_labels(threads_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary escalation proxy labels to thread DataFrame.

    Escalation = 1 if:
    - Contains escalation phrases, OR
    - No brand reply (unanswered), OR
    - Excessive conversation length (> 90th percentile)
    """
    print("Adding escalation proxy labels ...")

    # Escalation phrase detection
    phrase_results = threads_df['customer_texts'].apply(detect_escalation_phrases)
    threads_df['has_escalation_phrase'] = phrase_results.apply(lambda x: x[0])
    threads_df['escalation_phrases_found'] = phrase_results.apply(lambda x: x[1])

    # Length threshold (90th percentile)
    length_threshold = threads_df['num_turns'].quantile(0.90)

    # Composite escalation label
    threads_df['escalation_needed'] = (
        threads_df['has_escalation_phrase'] |
        (~threads_df['has_brand_reply']) |
        (threads_df['num_turns'] > length_threshold)
    ).astype(int)

    esc_rate = threads_df['escalation_needed'].mean()
    print(f"  Escalation rate: {esc_rate:.1%}")
    print(f"    - Phrase-based: {threads_df['has_escalation_phrase'].mean():.1%}")
    print(f"    - No reply: {(~threads_df['has_brand_reply']).mean():.1%}")
    print(f"    - Long thread (>{length_threshold:.0f} turns): "
          f"{(threads_df['num_turns'] > length_threshold).mean():.1%}")

    return threads_df


def prepare_twitter_features(threads_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute conversation-level features for the bandit from Twitter threads.

    Features:
    - num_turns: conversation length
    - num_customer_msgs, num_brand_msgs
    - avg_msg_length: average message length
    - has_question: contains '?'
    - urgency_score: count of urgency indicators
    - text_length: total customer text length
    """
    df = threads_df.copy()

    # Text length features
    df['avg_msg_length'] = df['customer_texts'].str.len() / df['num_customer_msgs'].clip(lower=1)
    df['text_length'] = df['customer_texts'].str.len()

    # Question detection
    df['has_question'] = df['customer_texts'].str.contains(r'\?', regex=True).astype(int)

    # Urgency indicators
    urgency_words = ['urgent', 'asap', 'immediately', 'emergency', 'critical', 'help']
    df['urgency_score'] = df['customer_texts'].str.lower().apply(
        lambda x: sum(1 for w in urgency_words if w in str(x))
    )

    # Exclamation count (frustration proxy)
    df['exclamation_count'] = df['customer_texts'].str.count('!')

    # Caps ratio (frustration proxy)
    df['caps_ratio'] = df['customer_texts'].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
    )

    print(f"  Computed {len(df.columns)} features for {len(df):,} threads")
    return df
