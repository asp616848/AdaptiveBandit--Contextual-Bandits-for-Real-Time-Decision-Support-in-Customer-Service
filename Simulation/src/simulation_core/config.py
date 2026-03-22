from pathlib import Path


def _resolve_project_root() -> Path:
    here = Path(__file__).resolve()

    # Typical layout: <project>/Simulation/src/simulation_core/config.py
    candidate = here.parents[3]
    if (candidate / "Simulation").exists():
        return candidate

    # Fallback: walk upward until we find folder containing Simulation.
    for parent in here.parents:
        if (parent / "Simulation").exists():
            return parent

    # Last resort keeps prior behavior for unusual layouts.
    return here.parents[4]


PROJECT_ROOT = _resolve_project_root()
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

# Domain-conditioned priors let us model realistic tier mix shifts.
TIER_PRIOR_BY_DOMAIN = {
    DOMAIN_TWITTER: {
        "Enterprise": 0.06,
        "Business+": 0.14,
        "Pro": 0.30,
        "Free": 0.50,
    },
    DOMAIN_OPENASSISTANT: {
        "Enterprise": 0.12,
        "Business+": 0.22,
        "Pro": 0.38,
        "Free": 0.28,
    },
}

TIER_VALUE_WEIGHT = {
    "Enterprise": 1.0,
    "Business+": 0.7,
    "Pro": 0.4,
    "Free": 0.1,
}
