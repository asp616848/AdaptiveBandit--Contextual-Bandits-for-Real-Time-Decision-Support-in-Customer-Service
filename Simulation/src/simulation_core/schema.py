from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class TurnRecord:
    conv_id: str
    turn_id: int
    domain_source: str
    speaker_role: str
    text: str
    text_length: int
    turn_index: int
    conv_length: int
    tier: Optional[str] = None
    tier_probs: Optional[Dict[str, float]] = None
    tier_confidence: Optional[float] = None
    customer_value_weight: Optional[float] = None
    action_label: Optional[str] = None
    action_confidence: Optional[float] = None
    action_probs: Optional[Dict[str, float]] = None
    sentiment_score: Optional[float] = None
    sentiment_confidence: Optional[float] = None
    frustration_score: Optional[float] = None
    frustration_confidence: Optional[float] = None
    annotator_confidence: Optional[float] = None
