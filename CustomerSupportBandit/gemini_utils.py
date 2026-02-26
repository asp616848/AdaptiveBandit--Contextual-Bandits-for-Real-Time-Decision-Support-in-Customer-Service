"""
Gemini API Integration for Customer Support Bandit.

Uses Google Gemini (gemini-2.0-flash) as an LLM-as-judge for:
1. Sentiment scoring — more nuanced than rule-based
2. Conversation quality assessment
3. Escalation risk classification
4. Simulated customer responses for environment enrichment

Requires: pip install google-genai
Set GEMINI_API_KEY environment variable or pass directly.
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple


def _get_client():
    """Lazy-load the Gemini client."""
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "google-genai not installed. Run: pip install google-genai"
        )

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
        )

    client = genai.Client(api_key=api_key)
    return client


# ═══════════════════════════════════════════════════════════
# 1. Sentiment Scoring
# ═══════════════════════════════════════════════════════════

_SENTIMENT_PROMPT = """\
You are a customer-support sentiment analyst. Rate the customer's sentiment
in the following message on a scale from -1.0 (extremely negative / angry)
to +1.0 (extremely positive / satisfied). Return ONLY a JSON object:
{{"score": <float>, "label": "<negative|neutral|positive>", "reasoning": "<one sentence>"}}

Customer message:
\"\"\"
{text}
\"\"\"
"""


def score_sentiment_gemini(text: str, client=None) -> Dict:
    """
    Score sentiment of a single message using Gemini.

    Returns dict with 'score' (float in [-1, 1]), 'label', 'reasoning'.
    """
    if client is None:
        client = _get_client()

    prompt = _SENTIMENT_PROMPT.format(text=text[:2000])

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        result = json.loads(response.text.strip().strip("```json").strip("```"))
        result['score'] = float(np.clip(result.get('score', 0.0), -1.0, 1.0))
        return result
    except Exception as e:
        return {'score': 0.0, 'label': 'neutral', 'reasoning': f'Error: {e}'}


def score_sentiment_batch(texts: List[str],
                           client=None,
                           delay: float = 0.1) -> List[Dict]:
    """Score sentiment for a batch of messages (with rate limiting)."""
    if client is None:
        client = _get_client()

    results = []
    for text in texts:
        result = score_sentiment_gemini(text, client)
        results.append(result)
        time.sleep(delay)  # Rate limit

    return results


# ═══════════════════════════════════════════════════════════
# 2. Conversation Quality Assessment
# ═══════════════════════════════════════════════════════════

_QUALITY_PROMPT = """\
You are evaluating the quality of a customer support conversation.
Assess the following conversation and return ONLY a JSON object:
{{
  "quality_score": <float 0.0 to 1.0>,
  "resolved": <true|false>,
  "customer_satisfaction": <float 1.0 to 5.0>,
  "escalation_needed": <true|false>,
  "complexity": "<simple|moderate|complex>",
  "reasoning": "<one sentence>"
}}

Conversation:
\"\"\"
{conversation}
\"\"\"
"""


def assess_conversation_quality(texts: List[str],
                                 roles: Optional[List[str]] = None,
                                 client=None) -> Dict:
    """
    Assess overall quality of a conversation using Gemini.

    Parameters
    ----------
    texts : list of str
        Messages in the conversation.
    roles : list of str, optional
        Role labels (e.g., 'customer', 'agent').

    Returns
    -------
    dict
        Quality assessment with score, resolution status, etc.
    """
    if client is None:
        client = _get_client()

    # Format conversation
    if roles is None:
        roles = ['Customer' if i % 2 == 0 else 'Agent' for i in range(len(texts))]

    conversation = "\n".join(
        f"{role}: {text}" for role, text in zip(roles, texts)
    )

    prompt = _QUALITY_PROMPT.format(conversation=conversation[:4000])

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        result = json.loads(response.text.strip().strip("```json").strip("```"))
        result['quality_score'] = float(
            np.clip(result.get('quality_score', 0.5), 0.0, 1.0)
        )
        return result
    except Exception as e:
        return {
            'quality_score': 0.5,
            'resolved': False,
            'customer_satisfaction': 3.0,
            'escalation_needed': False,
            'complexity': 'moderate',
            'reasoning': f'Error: {e}',
        }


# ═══════════════════════════════════════════════════════════
# 3. Escalation Risk Classification
# ═══════════════════════════════════════════════════════════

_ESCALATION_PROMPT = """\
You are a customer support routing system. Given the following customer
message(s), estimate the probability that this conversation requires
escalation to a human agent (vs. being handled by an automated bot).

Consider: emotional intensity, technical complexity, account/billing issues,
explicit requests for human help, legal threats, churn signals.

Return ONLY a JSON object:
{{
  "escalation_probability": <float 0.0 to 1.0>,
  "risk_level": "<low|medium|high|critical>",
  "signals": [<list of detected risk signals>],
  "recommended_action": "<bot|human>"
}}

Customer message(s):
\"\"\"
{text}
\"\"\"
"""


def classify_escalation_risk(texts: List[str],
                              client=None) -> Dict:
    """
    Classify escalation risk using Gemini LLM-as-judge.

    Parameters
    ----------
    texts : list of str
        Customer messages.

    Returns
    -------
    dict
        Escalation risk assessment.
    """
    if client is None:
        client = _get_client()

    combined = "\n---\n".join(texts[:5])  # Limit to 5 messages
    prompt = _ESCALATION_PROMPT.format(text=combined[:3000])

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        result = json.loads(response.text.strip().strip("```json").strip("```"))
        result['escalation_probability'] = float(
            np.clip(result.get('escalation_probability', 0.5), 0.0, 1.0)
        )
        return result
    except Exception as e:
        return {
            'escalation_probability': 0.5,
            'risk_level': 'medium',
            'signals': [],
            'recommended_action': 'bot',
            'error': str(e),
        }


# ═══════════════════════════════════════════════════════════
# 4. Simulated Customer Response Generator
# ═══════════════════════════════════════════════════════════

_SIM_CUSTOMER_PROMPT = """\
You are simulating a {tier}-tier SaaS customer in a support conversation.
The customer's frustration level is {frustration}/10.
The issue complexity is: {complexity}.

Given the conversation so far, generate the customer's NEXT message.
Keep it realistic — 1-3 sentences, natural language.
Do NOT include any JSON or formatting — just the customer's message text.

Conversation so far:
{history}

Customer's next message:"""


def simulate_customer_response(conversation_history: List[Dict],
                                tier: str = "Pro",
                                frustration: int = 5,
                                complexity: str = "moderate",
                                client=None) -> str:
    """
    Generate a simulated customer response using Gemini.

    Used to enrich the environment with more realistic multi-turn dynamics.

    Parameters
    ----------
    conversation_history : list of dict
        Previous messages with 'role' and 'text' keys.
    tier : str
        Customer tier (affects communication style).
    frustration : int
        Frustration level 1-10.
    complexity : str
        Issue complexity: simple/moderate/complex.

    Returns
    -------
    str
        Simulated customer message.
    """
    if client is None:
        client = _get_client()

    history = "\n".join(
        f"{msg['role']}: {msg['text']}" for msg in conversation_history[-6:]
    )

    prompt = _SIM_CUSTOMER_PROMPT.format(
        tier=tier,
        frustration=frustration,
        complexity=complexity,
        history=history,
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        # Fallback to template
        if frustration > 7:
            return "This still isn't working. I need this resolved immediately."
        elif frustration > 4:
            return "Can you provide an update on this issue?"
        else:
            return "Thanks, let me try that."


# ═══════════════════════════════════════════════════════════
# 5. Batch Processing Utilities
# ═══════════════════════════════════════════════════════════

def enrich_conversations_with_gemini(conversations: List[Dict],
                                       max_conversations: int = 100,
                                       client=None,
                                       delay: float = 0.2) -> List[Dict]:
    """
    Enrich conversation data with Gemini-powered features.

    Adds LLM-derived sentiment, quality, and escalation risk scores
    to each conversation dict.

    Parameters
    ----------
    conversations : list of dict
        Conversations with 'texts' key.
    max_conversations : int
        Limit for API calls.
    client : optional
        Gemini client.
    delay : float
        Delay between API calls (rate limiting).

    Returns
    -------
    list of dict
        Enriched conversations.
    """
    if client is None:
        client = _get_client()

    enriched = []
    n = min(len(conversations), max_conversations)

    print(f"Enriching {n} conversations with Gemini LLM-as-judge ...")

    for i, conv in enumerate(conversations[:n]):
        texts = conv.get('texts', [])
        if not texts:
            enriched.append(conv)
            continue

        # Sentiment of last customer message
        customer_texts = [t for j, t in enumerate(texts) if j % 2 == 0]
        if customer_texts:
            sent = score_sentiment_gemini(customer_texts[-1], client)
            conv['gemini_sentiment'] = sent.get('score', 0.0)
            conv['gemini_sentiment_label'] = sent.get('label', 'neutral')

        # Escalation risk
        esc = classify_escalation_risk(customer_texts[:3], client)
        conv['gemini_escalation_prob'] = esc.get('escalation_probability', 0.5)
        conv['gemini_risk_level'] = esc.get('risk_level', 'medium')
        conv['gemini_signals'] = esc.get('signals', [])

        enriched.append(conv)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{n} conversations")

        time.sleep(delay)

    print(f"  Done. Enriched {len(enriched)} conversations.")
    return enriched


def gemini_available() -> bool:
    """Check if Gemini API is configured and available."""
    try:
        _get_client()
        return True
    except (ImportError, ValueError):
        return False
