from __future__ import annotations

"""
LLM-based intent classifier for Stage 3 NLP observations.

Uses Ollama (same endpoint as NLGLayer) to classify customer intent from the
current conversation history and RAG policy context.  Returns a compact dict
of features that the NLPObservationWrapper converts into a 9D obs vector.

The classifier uses a small, deterministic (temperature=0) prompt so that
the same conversation always maps to the same intent features.  A simple
in-process cache keyed on conversation text avoids redundant LLM calls.

Supported models (set via env var SUPPORT_SIM_INTENT_MODEL):
  - phi3              (default; 3.8B, ~7 GB VRAM — best quality)
  - tinyllama         (1.1B, ~2 GB — fastest)
  - llama3            (8B, ~15 GB — highest accuracy, if VRAM allows)
"""

import hashlib
import json
import os
import re
from typing import Any

from openai import OpenAI


# Ordered list of canonical intents derived from the 55 ABCD subflows.
# The index is used as the normalised intent_id feature.
INTENT_LABELS = [
    "account_access",       # 0 — password reset, 2fa, login issues
    "billing_dispute",      # 1 — charges, refunds, invoices
    "order_status",         # 2 — shipping, delivery, tracking
    "product_issue",        # 3 — defect, wrong item, damage
    "subscription",         # 4 — plan change, cancellation, upgrade
    "technical_support",    # 5 — setup, configuration, error
    "general_inquiry",      # 6 — information, pricing, features
    "complaint",            # 7 — dissatisfied, escalation-prone
    "other",                # 8 — anything not captured above
]

SENTIMENT_LABELS = ["frustrated", "neutral", "satisfied"]

# Action index → label (must match SupportEnv.ACTION_NAMES)
ACTION_LABELS = ["AskInfo", "ProvideSolution", "AffectiveRepair", "Escalate", "Close"]


_SYSTEM_PROMPT = """\
You are an expert customer support triage system.
Given a customer support conversation, classify the current state.
Return ONLY a JSON object — no explanation, no markdown fences.

JSON schema:
{
  "intent":            <one of: account_access | billing_dispute | order_status |
                         product_issue | subscription | technical_support |
                         general_inquiry | complaint | other>,
  "confidence":        <float 0.0-1.0, how confident you are in the intent>,
  "sentiment":         <one of: frustrated | neutral | satisfied>,
  "suggested_action":  <one of: AskInfo | ProvideSolution | AffectiveRepair | Escalate | Close>,
  "escalation_needed": <true | false — true only if human escalation is clearly required>,
  "info_completeness": <float 0.0-1.0, fraction of customer information already provided>
}

Rules:
- escalation_needed = true only when the issue is technically unsolvable by chat, or the
  customer is severely distressed.
- Do not suggest Escalate unless escalation_needed is true.
- info_completeness = 1.0 when you have all info needed to attempt a solution.
"""


def _conversation_key(conversation_history: list[dict[str, str]], policy_context: str) -> str:
    text = json.dumps(conversation_history[-6:]) + policy_context[:200]
    return hashlib.md5(text.encode()).hexdigest()


class IntentClassifier:
    """Classify customer intent from conversation history via a small LLM."""

    def __init__(self, model: str | None = None, endpoint: str | None = None):
        self.model = model or os.getenv("SUPPORT_SIM_INTENT_MODEL", "phi3")
        self.endpoint = endpoint or os.getenv("SUPPORT_SIM_LLM_ENDPOINT", "http://localhost:11434/v1")
        self._cache: dict[str, dict[str, Any]] = {}
        try:
            self.client = OpenAI(base_url=self.endpoint, api_key="ollama")
        except Exception:
            self.client = None

    def is_available(self) -> bool:
        return self.client is not None

    def _fallback(self) -> dict[str, Any]:
        """Return a neutral, low-confidence classification when LLM is unavailable."""
        return {
            "intent": "general_inquiry",
            "confidence": 0.5,
            "sentiment": "neutral",
            "suggested_action": "AskInfo",
            "escalation_needed": False,
            "info_completeness": 0.3,
        }

    def _parse_response(self, text: str) -> dict[str, Any]:
        """Extract JSON from LLM output, tolerating markdown fences."""
        # Strip markdown fences if present
        text = re.sub(r"```[a-z]*", "", text).strip().strip("`").strip()
        # Find the first {...} block
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return {}
        return data

    def _validate(self, data: dict[str, Any]) -> dict[str, Any]:
        """Clamp and validate parsed fields; fill missing keys with defaults."""
        fallback = self._fallback()

        intent = str(data.get("intent", fallback["intent"]))
        if intent not in INTENT_LABELS:
            intent = "general_inquiry"

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        sentiment = str(data.get("sentiment", "neutral"))
        if sentiment not in SENTIMENT_LABELS:
            sentiment = "neutral"

        suggested_action = str(data.get("suggested_action", "AskInfo"))
        if suggested_action not in ACTION_LABELS:
            suggested_action = "AskInfo"

        escalation_needed = bool(data.get("escalation_needed", False))

        info_completeness = float(data.get("info_completeness", 0.3))
        info_completeness = max(0.0, min(1.0, info_completeness))

        return {
            "intent": intent,
            "confidence": confidence,
            "sentiment": sentiment,
            "suggested_action": suggested_action,
            "escalation_needed": escalation_needed,
            "info_completeness": info_completeness,
        }

    def classify(
        self,
        conversation_history: list[dict[str, str]],
        policy_context: str = "",
        subflow: str = "",
    ) -> dict[str, Any]:
        """Classify the current conversation state.

        Returns a validated dict with keys:
            intent, confidence, sentiment, suggested_action,
            escalation_needed, info_completeness
        """
        if not self.client:
            return self._fallback()

        cache_key = _conversation_key(conversation_history, policy_context)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build the user message from the last 6 turns.
        turns = conversation_history[-6:] if conversation_history else []
        if not turns:
            result = self._fallback()
            self._cache[cache_key] = result
            return result

        conv_text = "\n".join(
            f"{'Agent' if m.get('role') == 'user' else 'Customer'}: {m.get('content', '')}"
            for m in turns
        )
        if subflow:
            conv_text = f"[Issue type: {subflow.replace('_', ' ')}]\n\n" + conv_text
        if policy_context:
            conv_text += f"\n\n[Policy context excerpt]: {policy_context[:400]}"

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": conv_text},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=200,
                temperature=0.0,  # deterministic for RL stability
            )
            raw = (response.choices[0].message.content or "").strip()
            parsed = self._parse_response(raw)
            result = self._validate(parsed) if parsed else self._fallback()
        except Exception:
            result = self._fallback()

        self._cache[cache_key] = result
        # Prevent unbounded growth (simple eviction: clear when > 4096 entries)
        if len(self._cache) > 4096:
            self._cache.clear()
            self._cache[cache_key] = result

        return result

    def to_feature_vector(self, classification: dict[str, Any]) -> list[float]:
        """Convert a classification dict to a list of 6 normalised floats.

        Order: [intent_norm, confidence, sentiment_norm, action_norm,
                escalation_flag, info_completeness]
        """
        intent_id = INTENT_LABELS.index(classification.get("intent", "general_inquiry"))
        intent_norm = float(intent_id) / max(len(INTENT_LABELS) - 1, 1)

        confidence = float(classification.get("confidence", 0.5))

        sentiment = classification.get("sentiment", "neutral")
        sentiment_id = SENTIMENT_LABELS.index(sentiment) if sentiment in SENTIMENT_LABELS else 1
        sentiment_norm = float(sentiment_id) / max(len(SENTIMENT_LABELS) - 1, 1)

        action = classification.get("suggested_action", "AskInfo")
        action_id = ACTION_LABELS.index(action) if action in ACTION_LABELS else 0
        action_norm = float(action_id) / max(len(ACTION_LABELS) - 1, 1)

        escalation_flag = 1.0 if classification.get("escalation_needed", False) else 0.0

        info_completeness = float(classification.get("info_completeness", 0.3))

        return [intent_norm, confidence, sentiment_norm, action_norm, escalation_flag, info_completeness]
