"""
Configuration and economic parameters for the Customer Support Bandit system.

Anchored in Slack's tiered pricing model as described in the AML proposal.
All monetary values in INR (Indian Rupees).
"""

import numpy as np

# ═══════════════════════════════════════════════════════════
# Tier Economics (per-user per-month, INR)
# ═══════════════════════════════════════════════════════════

TIER_CONFIG = {
    "Free": {
        "monthly_cost": 20,
        "monthly_profit": -20,
        "margin": 0.0,
        "escalation_cost": 0,          # No escalation budget for free
        "csa_cost_per_user": 0,
        "churn_prob_bad_exp": 0.15,     # High churn but low CLV
        "clv": 0,                       # Network effect value only
        "csat_floor": 3.0,
        "driver": "network_effect",
        "max_escalation_rate": 0.02,    # Minimal human involvement
    },
    "Pro": {
        "monthly_cost": 77,
        "monthly_profit": 168,
        "margin": 0.68,
        "escalation_cost": 15,
        "csa_cost_per_user": 15,
        "churn_prob_bad_exp": 0.08,
        "clv": 168 * 24,               # ~2 year CLV
        "csat_floor": 3.5,
        "driver": "volume_cash_cow",
        "max_escalation_rate": 0.15,
    },
    "Business+": {
        "monthly_cost": 200,
        "monthly_profit": 357,
        "margin": 0.64,
        "escalation_cost": 50,
        "csa_cost_per_user": 50,
        "churn_prob_bad_exp": 0.03,
        "clv": 357 * 36,               # ~3 year CLV
        "csat_floor": 4.0,
        "driver": "retention",
        "max_escalation_rate": 0.30,
    },
    "Enterprise": {
        "monthly_cost": 400,
        "monthly_profit": 600,
        "margin": 0.64,
        "escalation_cost": 100,
        "csa_cost_per_user": 100,
        "churn_prob_bad_exp": 0.02,
        "clv": 600 * 48,               # ~4 year CLV
        "csat_floor": 4.5,
        "driver": "retention_critical",
        "max_escalation_rate": 0.50,
    },
}

TIER_NAMES = list(TIER_CONFIG.keys())
TIER_TO_IDX = {t: i for i, t in enumerate(TIER_NAMES)}

# ═══════════════════════════════════════════════════════════
# Inference / Operational Costs
# ═══════════════════════════════════════════════════════════

BOT_INFERENCE_COST = 0.80       # INR per conversation (bandit routing)
FULL_RL_INFERENCE_COST = 2.50   # INR per conversation (full RL + LLM)
CSA_HOURLY_COST = 250.0         # INR per hour for a human CSA
AVG_HANDLE_TIME_MIN = 8.0       # Average minutes per escalated conversation

# Capacity constraint
K_HUMAN_SLOTS_PER_HOUR = 50     # Max human escalations per hour

# ═══════════════════════════════════════════════════════════
# Actions
# ═══════════════════════════════════════════════════════════

# Contextual Bandit actions (routing decision)
BANDIT_ACTIONS = ["bot", "human"]
NUM_BANDIT_ACTIONS = len(BANDIT_ACTIONS)

# Full RL MDP actions
MDP_ACTIONS = ["ask_info", "provide_solution", "escalate", "close"]
NUM_MDP_ACTIONS = len(MDP_ACTIONS)

# ═══════════════════════════════════════════════════════════
# Reward Weights (Equation from RL Proposal)
# ═══════════════════════════════════════════════════════════

REWARD_WEIGHTS = {
    "alpha_sentiment": 1.0,         # Weight for sentiment improvement
    "beta_resolution": 2.0,         # Weight for resolution probability
    "gamma_cost_saved": 1.5,        # Weight for cost saving
    "turn_penalty": 0.1,            # Penalty per additional turn
    "resolution_bonus": 5.0,        # Bonus for successful resolution
    "escalation_penalty_scale": 1.0,# Scaled by tier escalation cost
    "churn_penalty_scale": 1.0,     # Scaled by tier churn cost
}

# ═══════════════════════════════════════════════════════════
# Feature Engineering
# ═══════════════════════════════════════════════════════════

# Conversation features
MAX_TURNS = 20
SENTIMENT_WINDOW = 3              # Turns for sentiment trajectory
TEXT_MAX_LEN = 512                # Max tokens for text encoding

# Escalation phrase indicators
ESCALATION_PHRASES = [
    "speak to manager", "talk to a person", "human agent",
    "not resolved", "still not working", "this is unacceptable",
    "cancel my subscription", "worst service", "terrible experience",
    "file a complaint", "escalate", "supervisor",
    "refund", "compensation", "legal action",
    "been waiting", "no response", "ignored",
]

# ═══════════════════════════════════════════════════════════
# Model Hyperparameters
# ═══════════════════════════════════════════════════════════

# LinUCB
LINUCB_ALPHA = 1.0               # Exploration parameter

# Thompson Sampling
TS_PRIOR_MEAN = 0.0
TS_PRIOR_VAR = 1.0

# DQN
DQN_CONFIG = {
    "hidden_dims": [128, 64, 32],
    "learning_rate": 1e-3,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 0.995,
    "batch_size": 64,
    "buffer_size": 50000,
    "target_update_freq": 100,
}

# Training
TRAINING_CONFIG = {
    "num_episodes": 5000,
    "eval_every": 100,
    "save_every": 500,
    "early_stop_patience": 10,
    "seed": 42,
}

# ═══════════════════════════════════════════════════════════
# Data Paths
# ═══════════════════════════════════════════════════════════

BASE_DIR = r'B:\College\RL\AdaptiveBandit--Contextual-Bandits-for-Real-Time-Decision-Support-in-Customer-Service'

DATA_PATHS = {
    "twitter": f'{BASE_DIR}\\twitter\\twcs\\twcs.csv',
    "openassistant_train": f'{BASE_DIR}\\OpenAssistant Conversations Dataset\\train.parquet',
    "openassistant_val": f'{BASE_DIR}\\OpenAssistant Conversations Dataset\\valid.parquet',
    "openassistant_train_csv": f'{BASE_DIR}\\OpenAssistant Conversations Dataset\\oasst1-train.csv',
    "openassistant_val_csv": f'{BASE_DIR}\\OpenAssistant Conversations Dataset\\oasst1-val.csv',
}

OUTPUT_DIR = f'{BASE_DIR}\\CustomerSupportBandit\\outputs'
WEIGHTS_DIR = f'{BASE_DIR}\\CustomerSupportBandit\\weights'
