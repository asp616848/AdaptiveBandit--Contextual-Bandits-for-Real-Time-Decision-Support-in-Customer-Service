"""
Feature engineering pipeline for the Customer Support Bandit.

Converts raw conversation data into standardized feature vectors suitable for
contextual bandit and RL agents. Features span three categories:
  1. Conversation state (sentiment, turn count, complexity)
  2. User context (tier, CLV proxy, history)
  3. Outcome prediction (resolution prob, frustration score, escalation risk)
"""

import numpy as np
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    TIER_CONFIG, TIER_NAMES, TIER_TO_IDX,
    ESCALATION_PHRASES, MAX_TURNS, SENTIMENT_WINDOW,
)


# ═══════════════════════════════════════════════════════════
# Simple Sentiment Scorer (rule-based, no external dependencies)
# ═══════════════════════════════════════════════════════════

# Positive / negative word lists for lightweight sentiment
_POSITIVE_WORDS = {
    'thank', 'thanks', 'great', 'good', 'excellent', 'awesome', 'perfect',
    'wonderful', 'amazing', 'love', 'appreciate', 'helpful', 'solved',
    'resolved', 'fixed', 'works', 'working', 'happy', 'pleased',
}

_NEGATIVE_WORDS = {
    'bad', 'terrible', 'awful', 'worst', 'horrible', 'angry', 'frustrated',
    'annoyed', 'disappointed', 'unhappy', 'broken', 'fail', 'failed',
    'error', 'bug', 'crash', 'slow', 'useless', 'waste', 'never',
    'unacceptable', 'ridiculous', 'pathetic', 'disgusting',
}


def compute_sentiment(text: str) -> float:
    """
    Simple rule-based sentiment score in [-1, 1].
    Positive values = positive sentiment.
    """
    if not isinstance(text, str) or len(text) == 0:
        return 0.0

    words = set(re.findall(r'\b\w+\b', text.lower()))
    pos_count = len(words & _POSITIVE_WORDS)
    neg_count = len(words & _NEGATIVE_WORDS)
    total = pos_count + neg_count

    if total == 0:
        return 0.0
    return (pos_count - neg_count) / total


def compute_sentiment_trajectory(texts: List[str],
                                  window: int = SENTIMENT_WINDOW) -> Dict:
    """
    Compute sentiment trajectory features over a conversation.

    Returns dict with:
    - sentiment_current: sentiment of last message
    - sentiment_mean: mean over all messages
    - sentiment_trend: slope of sentiment over last `window` messages
    - sentiment_volatility: std of sentiment scores
    - sentiment_min: worst sentiment observed
    """
    if not texts:
        return {
            'sentiment_current': 0.0,
            'sentiment_mean': 0.0,
            'sentiment_trend': 0.0,
            'sentiment_volatility': 0.0,
            'sentiment_min': 0.0,
        }

    scores = [compute_sentiment(t) for t in texts]
    recent = scores[-window:] if len(scores) >= window else scores

    # Trend: linear slope
    if len(recent) >= 2:
        x = np.arange(len(recent))
        trend = np.polyfit(x, recent, 1)[0]
    else:
        trend = 0.0

    return {
        'sentiment_current': scores[-1],
        'sentiment_mean': np.mean(scores),
        'sentiment_trend': trend,
        'sentiment_volatility': np.std(scores) if len(scores) > 1 else 0.0,
        'sentiment_min': np.min(scores),
    }


# ═══════════════════════════════════════════════════════════
# Complexity & Content Features
# ═══════════════════════════════════════════════════════════

def compute_complexity_features(texts: List[str]) -> Dict:
    """
    Compute text complexity features from a conversation.

    Returns dict with:
    - avg_word_count: average words per message
    - max_word_count: longest message
    - total_word_count: total words
    - avg_sentence_length: average sentence length
    - technical_term_density: fraction of technical terms
    - has_error_code: whether text contains error code patterns
    - has_url: whether text contains URLs
    """
    if not texts:
        return {k: 0.0 for k in [
            'avg_word_count', 'max_word_count', 'total_word_count',
            'avg_sentence_length', 'technical_term_density',
            'has_error_code', 'has_url',
        ]}

    word_counts = [len(t.split()) for t in texts]
    all_text = ' '.join(texts).lower()
    words = re.findall(r'\b\w+\b', all_text)

    # Technical terms
    tech_terms = {'error', 'bug', 'crash', 'api', 'server', 'timeout',
                  'database', 'ssl', 'dns', 'http', 'port', 'config',
                  'deploy', 'debug', 'stack', 'trace', 'exception',
                  'authentication', 'token', 'permission', 'endpoint'}
    tech_count = sum(1 for w in words if w in tech_terms)

    # Error code pattern (e.g., ERR-404, Error 500)
    has_error_code = bool(re.search(
        r'(?:error|err|code|status)\s*[:#-]?\s*\d{3,}', all_text
    ))

    # URL detection
    has_url = bool(re.search(r'https?://|www\.', all_text))

    return {
        'avg_word_count': np.mean(word_counts),
        'max_word_count': np.max(word_counts),
        'total_word_count': sum(word_counts),
        'avg_sentence_length': np.mean(word_counts),
        'technical_term_density': tech_count / max(len(words), 1),
        'has_error_code': int(has_error_code),
        'has_url': int(has_url),
    }


def compute_escalation_features(texts: List[str]) -> Dict:
    """
    Compute escalation-risk features.

    Returns dict with:
    - escalation_phrase_count: number of escalation phrases detected
    - has_escalation_phrase: binary indicator
    - frustration_score: cumulative frustration estimate
    """
    all_text = ' '.join(texts).lower() if texts else ''

    phrase_count = sum(1 for p in ESCALATION_PHRASES if p in all_text)

    # Cumulative frustration: f(t) = f(t-1) + alpha * (1 - delta_sentiment)
    alpha = 0.3
    frustration = 0.0
    for t in texts:
        sent = compute_sentiment(t)
        frustration = frustration + alpha * (1 - sent)

    return {
        'escalation_phrase_count': phrase_count,
        'has_escalation_phrase': int(phrase_count > 0),
        'frustration_score': frustration,
    }


# ═══════════════════════════════════════════════════════════
# Tier Assignment (simulated for public data)
# ═══════════════════════════════════════════════════════════

def assign_simulated_tier(rng: np.random.RandomState = None) -> str:
    """
    Assign a simulated SaaS tier to a conversation.

    Distribution based on typical SaaS user mix:
    - Free: 60%, Pro: 25%, Business+: 12%, Enterprise: 3%
    """
    if rng is None:
        rng = np.random.RandomState()
    return rng.choice(TIER_NAMES, p=[0.60, 0.25, 0.12, 0.03])


def get_tier_features(tier: str) -> Dict:
    """Get economic features for a given tier."""
    cfg = TIER_CONFIG[tier]
    return {
        'tier_idx': TIER_TO_IDX[tier],
        'tier_profit': cfg['monthly_profit'],
        'tier_escalation_cost': cfg['escalation_cost'],
        'tier_churn_prob': cfg['churn_prob_bad_exp'],
        'tier_clv': cfg['clv'],
        'tier_csat_floor': cfg['csat_floor'],
    }


# ═══════════════════════════════════════════════════════════
# Full Feature Vector Builder
# ═══════════════════════════════════════════════════════════

def build_feature_vector(texts: List[str],
                          tier: str,
                          turn_number: int,
                          max_turns: int = MAX_TURNS) -> np.ndarray:
    """
    Build a complete feature vector for a conversation state.

    Combines:
    - Sentiment features (5)
    - Complexity features (7)
    - Escalation features (3)
    - Tier features (6)
    - Turn features (2)

    Total: 23-dimensional feature vector.

    Parameters
    ----------
    texts : list of str
        Messages in the conversation so far.
    tier : str
        Customer tier name.
    turn_number : int
        Current turn number (0-indexed).
    max_turns : int
        Maximum expected turns (for normalization).

    Returns
    -------
    np.ndarray
        Feature vector of shape (23,).
    """
    # Sentiment features
    sent_feats = compute_sentiment_trajectory(texts)

    # Complexity features
    complexity_feats = compute_complexity_features(texts)

    # Escalation features
    esc_feats = compute_escalation_features(texts)

    # Tier features
    tier_feats = get_tier_features(tier)

    # Turn features
    turn_feats = {
        'turn_normalized': turn_number / max_turns,
        'turn_count': min(turn_number, max_turns),
    }

    # Combine into ordered vector
    feature_values = []
    for d in [sent_feats, complexity_feats, esc_feats, tier_feats, turn_feats]:
        feature_values.extend(d.values())

    return np.array(feature_values, dtype=np.float32)


def get_feature_names() -> List[str]:
    """Return ordered list of feature names matching build_feature_vector output."""
    names = []

    # Sentiment
    names.extend(['sentiment_current', 'sentiment_mean', 'sentiment_trend',
                  'sentiment_volatility', 'sentiment_min'])

    # Complexity
    names.extend(['avg_word_count', 'max_word_count', 'total_word_count',
                  'avg_sentence_length', 'technical_term_density',
                  'has_error_code', 'has_url'])

    # Escalation
    names.extend(['escalation_phrase_count', 'has_escalation_phrase',
                  'frustration_score'])

    # Tier
    names.extend(['tier_idx', 'tier_profit', 'tier_escalation_cost',
                  'tier_churn_prob', 'tier_clv', 'tier_csat_floor'])

    # Turn
    names.extend(['turn_normalized', 'turn_count'])

    return names


FEATURE_DIM = len(get_feature_names())  # 23
