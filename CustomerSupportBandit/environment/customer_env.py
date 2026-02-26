"""
Customer Support Environment — Gym-like interface.

Simulates tiered SaaS customer support conversations using real data
features. Supports both:
  - Bandit mode: single routing decision (bot vs human)
  - MDP mode: multi-turn dialogue (ask, solve, escalate, close)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    TIER_CONFIG, TIER_NAMES, TIER_TO_IDX,
    BANDIT_ACTIONS, MDP_ACTIONS, NUM_BANDIT_ACTIONS, NUM_MDP_ACTIONS,
    MAX_TURNS, K_HUMAN_SLOTS_PER_HOUR,
    REWARD_WEIGHTS, BOT_INFERENCE_COST,
)
from data.feature_engineer import (
    build_feature_vector, assign_simulated_tier,
    compute_sentiment, FEATURE_DIM,
)


class CustomerSupportEnv:
    """
    Gym-like environment for customer support routing/dialogue.

    Two modes:
    1. BANDIT: Agent makes a single routing decision per conversation.
       Actions = {0: bot, 1: human}
       One step per episode.

    2. MDP: Agent manages a multi-turn dialogue.
       Actions = {0: ask_info, 1: provide_solution, 2: escalate, 3: close}
       Multiple steps per episode until terminal.
    """

    def __init__(self, mode: str = "bandit",
                 conversations: Optional[List[Dict]] = None,
                 capacity_k: int = K_HUMAN_SLOTS_PER_HOUR,
                 seed: int = 42):
        """
        Parameters
        ----------
        mode : str
            "bandit" or "mdp"
        conversations : list of dict, optional
            Pre-loaded conversation data. Each dict should have:
            - 'texts': list of message strings
            - 'tier': tier name (or None to simulate)
            - 'escalation_needed': ground truth (0/1)
            - 'num_turns': number of turns
        capacity_k : int
            Max human escalations per evaluation hour.
        seed : int
            Random seed.
        """
        assert mode in ("bandit", "mdp"), f"Invalid mode: {mode}"
        self.mode = mode
        self.capacity_k = capacity_k
        self.rng = np.random.RandomState(seed)

        # Load or create conversations
        if conversations is not None:
            self.conversations = conversations
        else:
            self.conversations = self._generate_synthetic_conversations(1000)

        self.n_conversations = len(self.conversations)
        self.current_idx = 0

        # Episode state
        self.current_conv = None
        self.current_turn = 0
        self.current_texts_seen = []
        self.done = False

        # Capacity tracking (per evaluation hour)
        self.human_slots_used = 0

        # Actions
        if mode == "bandit":
            self.n_actions = NUM_BANDIT_ACTIONS
            self.action_names = BANDIT_ACTIONS
        else:
            self.n_actions = NUM_MDP_ACTIONS
            self.action_names = MDP_ACTIONS

        self.feature_dim = FEATURE_DIM

    def reset(self) -> np.ndarray:
        """
        Reset to a new conversation episode.

        Returns
        -------
        np.ndarray
            Initial observation (feature vector).
        """
        # Pick next conversation
        conv = self.conversations[self.current_idx % self.n_conversations]
        self.current_idx += 1

        self.current_conv = conv
        self.current_turn = 0
        self.done = False

        # Show first message
        texts = conv.get('texts', ['Hello, I need help.'])
        self.current_texts_seen = [texts[0]] if texts else ['']

        tier = conv.get('tier', assign_simulated_tier(self.rng))
        self.current_tier = tier

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take an action in the environment.

        Parameters
        ----------
        action : int
            Action index.

        Returns
        -------
        observation : np.ndarray
            Next state features.
        reward : float
            Economic reward.
        done : bool
            Whether episode is finished.
        info : dict
            Additional information.
        """
        assert not self.done, "Episode is done. Call reset()."

        if self.mode == "bandit":
            return self._step_bandit(action)
        else:
            return self._step_mdp(action)

    def _step_bandit(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Single-step bandit: route to bot (0) or human (1)."""
        conv = self.current_conv
        tier = self.current_tier
        cfg = TIER_CONFIG[tier]

        escalation_needed = conv.get('escalation_needed', 0)
        quality = conv.get('avg_quality', 0.5)

        # Compute reward based on routing decision
        reward, info = self._compute_bandit_reward(
            action, escalation_needed, tier, quality
        )

        # Track capacity
        if action == 1:  # human
            self.human_slots_used += 1

        self.done = True
        obs = self._get_observation()
        info['tier'] = tier
        info['action_name'] = BANDIT_ACTIONS[action]
        info['escalation_needed'] = escalation_needed
        info['capacity_used'] = self.human_slots_used

        return obs, reward, True, info

    def _step_mdp(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Multi-turn MDP step."""
        conv = self.current_conv
        tier = self.current_tier
        cfg = TIER_CONFIG[tier]
        texts = conv.get('texts', [])

        self.current_turn += 1

        # Show next message if available
        if self.current_turn < len(texts):
            self.current_texts_seen.append(texts[self.current_turn])

        # Compute reward and determine if done
        reward, info = self._compute_mdp_reward(action, tier)

        # Terminal conditions
        if action == 2:  # escalate
            self.done = True
            self.human_slots_used += 1
        elif action == 3:  # close
            self.done = True
        elif self.current_turn >= MAX_TURNS:
            self.done = True
            reward -= 2.0  # Penalty for exceeding max turns

        obs = self._get_observation()
        info['tier'] = tier
        info['action_name'] = MDP_ACTIONS[action]
        info['turn'] = self.current_turn
        info['done'] = self.done

        return obs, reward, self.done, info

    def _compute_bandit_reward(self, action: int, escalation_needed: int,
                                tier: str, quality: float) -> Tuple[float, Dict]:
        """
        Compute economic reward for bandit routing decision.

        reward = alpha * delta_sent + beta * delta_P(res) + gamma * cost_saved + r_term

        Simplified to:
        - Correct routing (bot when easy, human when needed): positive reward
        - Wrong routing: negative reward scaled by tier economics
        """
        cfg = TIER_CONFIG[tier]
        w = REWARD_WEIGHTS

        info = {}

        if action == 0:  # Bot
            if escalation_needed == 0:
                # Correct: bot handles easy case
                cost_saved = cfg['escalation_cost']  # Saved by not escalating
                satisfaction = 0.8 * quality
                reward = (w['gamma_cost_saved'] * cost_saved / 50.0 +
                         w['alpha_sentiment'] * satisfaction +
                         w['beta_resolution'] * 1.0)
                info['outcome'] = 'correct_deflection'
            else:
                # Wrong: bot handles hard case → unhappy customer
                churn_risk = cfg['churn_prob_bad_exp']
                churn_cost = churn_risk * cfg['clv']
                reward = (-w['churn_penalty_scale'] * churn_cost / 1000.0 -
                         w['alpha_sentiment'] * 0.5)
                info['outcome'] = 'missed_escalation'
        else:  # Human
            if escalation_needed == 1:
                # Correct: human handles hard case
                satisfaction = 0.9
                reward = (w['alpha_sentiment'] * satisfaction +
                         w['beta_resolution'] * 1.5 -
                         w['escalation_penalty_scale'] * cfg['escalation_cost'] / 100.0)
                info['outcome'] = 'correct_escalation'
            else:
                # Wrong: wasted human on easy case
                cost_wasted = cfg['escalation_cost']
                reward = (-w['escalation_penalty_scale'] * cost_wasted / 50.0 +
                         w['alpha_sentiment'] * 0.3)  # Still satisfied, just costly
                info['outcome'] = 'unnecessary_escalation'

        info['reward'] = reward
        return reward, info

    def _compute_mdp_reward(self, action: int,
                             tier: str) -> Tuple[float, Dict]:
        """
        Compute step reward for MDP dialogue actions.

        R_t = alpha * 1{Resolved} - C_esc * 1{Escalate}
              - L_churn * 1{Failure} - delta * TurnPenalty
        """
        cfg = TIER_CONFIG[tier]
        w = REWARD_WEIGHTS
        info = {}

        # Base turn penalty
        reward = -w['turn_penalty']

        # Sentiment improvement from current texts
        if len(self.current_texts_seen) >= 2:
            sent_now = compute_sentiment(self.current_texts_seen[-1])
            sent_prev = compute_sentiment(self.current_texts_seen[-2])
            delta_sent = sent_now - sent_prev
            reward += w['alpha_sentiment'] * delta_sent * 0.5

        if action == 0:  # ask_info
            # Information gathering: small positive for early turns
            if self.current_turn <= 3:
                reward += 0.1
            else:
                reward -= 0.05  # Diminishing returns

        elif action == 1:  # provide_solution
            # Resolution attempt
            conv = self.current_conv
            esc_needed = conv.get('escalation_needed', 0)
            if esc_needed == 0:
                # Easy problem: likely resolved
                reward += w['resolution_bonus'] * 0.7
            else:
                # Hard problem: partial resolution
                reward += w['resolution_bonus'] * 0.2

        elif action == 2:  # escalate
            # Hand off to human
            cost = cfg['escalation_cost']
            reward -= w['escalation_penalty_scale'] * cost / 50.0
            reward += w['beta_resolution'] * 0.8  # Human likely resolves

        elif action == 3:  # close
            # Close conversation
            conv = self.current_conv
            esc_needed = conv.get('escalation_needed', 0)
            if esc_needed == 0:
                reward += w['resolution_bonus'] * 0.5  # Good close
            else:
                # Premature close on hard problem → churn risk
                churn_cost = cfg['churn_prob_bad_exp'] * cfg['clv']
                reward -= w['churn_penalty_scale'] * churn_cost / 1000.0

        info['reward_components'] = {
            'action': MDP_ACTIONS[action],
            'turn': self.current_turn,
        }
        return reward, info

    def _get_observation(self) -> np.ndarray:
        """Build feature vector from current conversation state."""
        return build_feature_vector(
            texts=self.current_texts_seen,
            tier=self.current_tier,
            turn_number=self.current_turn,
        )

    def _generate_synthetic_conversations(self, n: int) -> List[Dict]:
        """
        Generate synthetic conversations for testing when no real data loaded.
        """
        conversations = []
        templates_easy = [
            "How do I reset my password?",
            "Where can I find my billing information?",
            "How do I change my notification settings?",
            "What are the keyboard shortcuts?",
            "How do I create a new channel?",
        ]
        templates_hard = [
            "Your service has been down for 3 hours and we're losing money!",
            "I've been waiting for a response for days. This is unacceptable!",
            "The API keeps returning 500 errors. Our production is broken.",
            "I want to speak to a manager immediately. Nothing has been resolved.",
            "We need to cancel our subscription unless this is fixed today.",
        ]

        for i in range(n):
            is_hard = self.rng.random() < 0.35
            tier = assign_simulated_tier(self.rng)

            if is_hard:
                base = self.rng.choice(templates_hard)
                follow_ups = [
                    "Still waiting...",
                    "This is really frustrating.",
                    "Can someone help?",
                    "I need this escalated.",
                ]
            else:
                base = self.rng.choice(templates_easy)
                follow_ups = [
                    "Thanks for the help!",
                    "That worked, appreciate it.",
                ]

            n_turns = min(self.rng.geometric(0.3) + 1, 15)
            texts = [base]
            for j in range(n_turns - 1):
                texts.append(self.rng.choice(follow_ups))

            conversations.append({
                'texts': texts,
                'tier': tier,
                'escalation_needed': int(is_hard),
                'num_turns': len(texts),
                'avg_quality': 0.3 if is_hard else 0.8,
                'conversation_id': i,
            })

        return conversations

    def get_capacity_info(self) -> Dict:
        """Return capacity utilization information."""
        return {
            'human_slots_used': self.human_slots_used,
            'capacity_k': self.capacity_k,
            'utilization': self.human_slots_used / max(self.capacity_k, 1),
            'slots_remaining': max(0, self.capacity_k - self.human_slots_used),
        }

    def reset_capacity(self):
        """Reset hourly capacity counter."""
        self.human_slots_used = 0

    @property
    def observation_dim(self) -> int:
        return self.feature_dim
