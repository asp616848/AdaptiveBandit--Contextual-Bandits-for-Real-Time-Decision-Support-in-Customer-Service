from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SIM_ROOT = PROJECT_ROOT / "Simulation"
ARTIFACTS_ROOT = SIM_ROOT / "artifacts"
REPORTS_ROOT = SIM_ROOT / "reports"

TWITTER_PATH = PROJECT_ROOT / "twitter" / "twcs" / "twcs.csv"
OPENASSISTANT_TRAIN_PATH = PROJECT_ROOT / "OpenAssistant Conversations Dataset" / "oasst1-train.csv"
OPENASSISTANT_VAL_PATH = PROJECT_ROOT / "OpenAssistant Conversations Dataset" / "oasst1-val.csv"

DOMAIN_TWITTER = "twitter_cs"
DOMAIN_OPENASSISTANT = "open_assistant"

ACTION_SPACE_7 = [
    "Ask_for_Information",
    "Provide_Solution",
    "Affective_Repair",
    "Escalate_to_Human",
    "Close_with_Feedback",
    "Proactive_Update",
    "Set_Expectation",
]

ACTION_SPACE_8 = ACTION_SPACE_7 + ["Unknown"]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTION_SPACE_7)}
ACTION_TO_ID_8 = {a: i for i, a in enumerate(ACTION_SPACE_8)}

LOW_CONFIDENCE_THRESHOLD = 0.65

TIER_VALUES = ["Enterprise", "Business+", "Pro", "Free"]
DEFAULT_TIER_DISTRIBUTION = [0.1, 0.2, 0.3, 0.4]
