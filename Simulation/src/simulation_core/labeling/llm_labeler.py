from __future__ import annotations

import os
from typing import Dict, List

from simulation_core.config import ACTION_SPACE_8
from simulation_core.utils.json_utils import safe_json_loads

try:
    from ollama import chat
except Exception:  # pragma: no cover - runtime optional
    chat = None


def _normalize_probs(raw: Dict[str, float]) -> Dict[str, float]:
    probs = {k: float(max(raw.get(k, 0.0), 0.0)) for k in ACTION_SPACE_8}
    total = sum(probs.values())
    if total <= 0:
        base = 1.0 / len(ACTION_SPACE_8)
        return {k: base for k in ACTION_SPACE_8}
    return {k: v / total for k, v in probs.items()}


def heuristic_action_probs(text: str) -> Dict[str, float]:
    del text
    # Keep fallback neutral to avoid brittle keyword rules.
    base = 1.0 / len(ACTION_SPACE_8)
    return {k: base for k in ACTION_SPACE_8}


def _require_llm() -> bool:
    # Strict by default: production labels should come from the configured LLM.
    return os.getenv("SIM_REQUIRE_LLM", "1").strip().lower() not in {"0", "false", "no"}


def _raise_llm_unavailable() -> None:
    raise RuntimeError(
        "LLM labeling is required but unavailable. Ensure Ollama is installed/running and the model is pulled. "
        "Set SIM_REQUIRE_LLM=0 only for debugging fallback behavior."
    )


def label_action_with_llm(text: str, model: str = "qwen3:4b") -> Dict[str, object]:
    heuristic = heuristic_action_probs(text)
    if chat is None:
        if _require_llm():
            _raise_llm_unavailable()
        label = max(heuristic, key=heuristic.get)
        return {
            "action_label": label,
            "action_probs": heuristic,
            "action_confidence": float(heuristic[label]),
            "annotator_confidence": float(heuristic[label]),
        }

    system = (
        "Classify the support AGENT utterance into exactly one label. "
        f"Labels: {ACTION_SPACE_8}. "
        "Return JSON with keys action_label and action_probs only. "
        "action_probs must include all labels and sum to 1."
    )

    payload = {"action_label": "Unknown", "action_probs": heuristic}
    try:
        response = chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text or ""},
            ],
            options={"temperature": 0.0},
        )
        parsed = safe_json_loads(response.message.content, payload)
        probs = _normalize_probs(parsed.get("action_probs", heuristic))
        label = parsed.get("action_label", "Unknown")
        if label not in ACTION_SPACE_8:
            label = max(probs, key=probs.get)
        confidence = float(max(probs.values()))
        return {
            "action_label": label,
            "action_probs": probs,
            "action_confidence": confidence,
            "annotator_confidence": confidence,
        }
    except Exception:
        if _require_llm():
            raise
        label = max(heuristic, key=heuristic.get)
        return {
            "action_label": label,
            "action_probs": heuristic,
            "action_confidence": 0.0,
            "annotator_confidence": 0.0,
        }


def score_customer_state_with_llm(text: str, model: str = "qwen3:4b") -> Dict[str, float]:
    # Fast fallback keeps pipeline deterministic when local model is unavailable.
    lowered = (text or "").lower()
    sentiment = 0.0
    frustration = 0.5

    neg = ["bad", "angry", "frustrat", "terrible", "not working", "cancel"]
    pos = ["thanks", "great", "resolved", "helpful", "perfect"]

    for token in neg:
        if token in lowered:
            sentiment -= 0.2
            frustration += 0.1
    for token in pos:
        if token in lowered:
            sentiment += 0.2
            frustration -= 0.1

    sentiment = max(-1.0, min(1.0, sentiment))
    frustration = max(0.0, min(1.0, frustration))

    if chat is None:
        if _require_llm():
            _raise_llm_unavailable()
        return {
            "sentiment_score": sentiment,
            "sentiment_confidence": 0.55,
            "frustration_score": frustration,
            "frustration_confidence": 0.55,
            "annotator_confidence": 0.55,
        }

    system = (
        "Score a CUSTOMER utterance with JSON only using keys sentiment_score, "
        "sentiment_confidence, frustration_score, frustration_confidence. "
        "Ranges: sentiment_score in [-1,1], confidence in [0,1], frustration in [0,1]."
    )
    payload = {
        "sentiment_score": sentiment,
        "sentiment_confidence": 0.0,
        "frustration_score": frustration,
        "frustration_confidence": 0.0,
    }

    try:
        response = chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text or ""},
            ],
            options={"temperature": 0.0},
        )
        parsed = safe_json_loads(response.message.content, payload)
        out = {
            "sentiment_score": float(max(-1.0, min(1.0, parsed.get("sentiment_score", sentiment)))),
            "sentiment_confidence": float(max(0.0, min(1.0, parsed.get("sentiment_confidence", 0.0)))),
            "frustration_score": float(max(0.0, min(1.0, parsed.get("frustration_score", frustration)))),
            "frustration_confidence": float(max(0.0, min(1.0, parsed.get("frustration_confidence", 0.0)))),
        }
        out["annotator_confidence"] = min(out["sentiment_confidence"], out["frustration_confidence"])
        return out
    except Exception:
        if _require_llm():
            raise
        return {
            "sentiment_score": sentiment,
            "sentiment_confidence": 0.0,
            "frustration_score": frustration,
            "frustration_confidence": 0.0,
            "annotator_confidence": 0.0,
        }
