"""
Phase II Environment — RL Prompt Selector
==========================================
Extended version of the Phase I CustomerServiceEnv.

Key differences from Phase I:
  * State is 15-dimensional (adds last-action one-hot, strategy usage history,
    repetition signal) — gives the MDP agent temporal context that bandits lacked.
  * Conversation transcript is maintained so the Mock LLM receives real
    conversational context (turn-by-turn utterance history).
  * transition_record() exposes the full (s, a, r, s', done) tuple for
    replay-buffer based algorithms (DQN).
  * Compatible with all three RL algorithms (DQN / A2C / PPO).

State vector (dim = 15):
  [0]  sentiment              — customer mood [0, 1]
  [1]  frustration            — customer frustration [0, 1]
  [2]  info_gathered          — information completeness [0, 1]
  [3]  turn_norm              — turn / max_turns  [0, 1]
  [4–8] last_action_onehot   — one-hot of most recent strategy (5 dims)
  [9–13] usage_counts_norm   — normalised count of each strategy used so far
  [14] repeat_signal         — 1 if same action chosen ≥ 3 times in a row, else 0
"""

import numpy as np
import random
import sys
import os

# allow importing from the parent contextual_bandits folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contextual_bandits'))

from strategy_prompts import STRATEGY_NAMES, NUM_STRATEGIES

# ── Customer utterance banks (used to build the conversation transcript) ────

_UTTERANCES = {
    # format: { action_that_was_taken : { outcome : [possible customer replies] } }
    "ask_info": {
        "positive": [
            "Sure, here are the details you asked for — I'm using the mobile app on iOS 17.",
            "Of course! It started happening after the last update, around three days ago.",
            "Good question — I'm on the premium plan and it's been happening every morning.",
        ],
        "negative": [
            "I already told you all of this! Why do I have to repeat myself again?",
            "I don't understand why you need more information — this should be simple to fix.",
            "Can we please just fix the issue? I've answered so many questions already.",
        ],
    },
    "provide_solution": {
        "positive": [
            "Oh that actually worked! Thank you so much, I really appreciate it.",
            "It's fixed now. I was so stressed about this — you're a lifesaver!",
            "Perfect, the steps you provided solved it completely. Very clear instructions.",
        ],
        "negative": [
            "I tried that already and it didn't work. What else can we try?",
            "Unfortunately that solution didn't resolve the issue at all.",
            "That made things worse actually. I'm really frustrated right now.",
        ],
    },
    "empathize": {
        "positive": [
            "Thank you for understanding — that actually makes me feel better.",
            "I appreciate you acknowledging that. It's been a really stressful situation.",
            "Thanks, I know you're trying to help. Let's work through this together.",
        ],
        "negative": [
            "I don't want apologies, I want my problem fixed as soon as possible.",
            "I've heard this before. I need actual solutions, not sympathy.",
            "It's a bit late for sorry — this has been going on for days.",
        ],
    },
    "escalate": {
        "positive": [
            "That's fine, as long as someone actually helps me. When will they contact me?",
            "Okay, I hope the specialist can fix this. I'll be expecting a call.",
            "Alright, thank you for being upfront about it. I'll wait to hear from them.",
        ],
        "negative": [
            "Why can't you just fix it now? I don't want to wait anymore.",
            "This is frustrating — I've already been passed around multiple times.",
            "Another transfer? This is taking way too long.",
        ],
    },
    "close": {
        "positive": [
            "Yes, everything is working perfectly now. Thank you!",
            "No, I think we've covered everything. Really appreciate your help today.",
            "All good now. You've been very helpful, I'll come back if anything else comes up.",
        ],
        "negative": [
            "Actually, wait — the issue is still there!",
            "Not quite — there's still one more thing I need help with.",
            "I wish this had been resolved sooner, but okay.",
        ],
    },
}

_OPENING_UTTERANCES = {
    "twitter": [
        "@support my account has been locked for 2 DAYS and nobody is helping me!! #frustrated",
        "Your app keeps crashing every time I try to check out — just lost my order AGAIN",
        "why is my subscription still showing as unpaid when I paid 3 days ago?? fix this",
    ],
    "reddit": [
        "Hey, I've been having a recurring issue with my account for about a week now. "
        "I've tried the obvious fixes but nothing has worked.",
        "Long-time user here. I'm running into a billing discrepancy and I'm not sure "
        "what's going on. Can someone help me sort this out?",
        "My service has been intermittently dropping for the past few days. "
        "Tech support sent me a troubleshooting guide but none of the steps resolved it.",
    ],
    "openassistant": [
        "Hi, I need some help with my account settings. "
        "I can't seem to find where to change my notification preferences.",
        "Hello! I'm having trouble accessing a feature that I believe I should have "
        "as part of my current plan. Could you look into this for me?",
        "Hi there. I received an error message this morning when trying to log in "
        "and now I'm locked out. What should I do?",
    ],
}


class MDPCustomerServiceEnv:
    """
    Phase II MDP environment.
    State dimension = 15.  Action space = 5 strategies.
    """

    ACTIONS     = STRATEGY_NAMES    # ['ask_info', 'provide_solution', ...]
    NUM_ACTIONS = NUM_STRATEGIES    # 5
    STATE_DIM   = 15

    PROFILES = {
        'twitter': {
            'max_turns':            10,
            'initial_sentiment':    0.30,
            'initial_frustration':  0.65,
            'resolution_threshold': 0.65,
            'info_gain_rate':       0.25,
            'sentiment_boost':      0.15,
            'name': 'Twitter Customer Support',
        },
        'reddit': {
            'max_turns':            20,
            'initial_sentiment':    0.50,
            'initial_frustration':  0.30,
            'resolution_threshold': 0.60,
            'info_gain_rate':       0.15,
            'sentiment_boost':      0.12,
            'name': 'Reddit DSTC8 Corpus',
        },
        'openassistant': {
            'max_turns':            15,
            'initial_sentiment':    0.55,
            'initial_frustration':  0.20,
            'resolution_threshold': 0.50,
            'info_gain_rate':       0.20,
            'sentiment_boost':      0.18,
            'name': 'OpenAssistant Conversations',
        },
    }

    def __init__(self, dataset: str = 'twitter'):
        if dataset not in self.PROFILES:
            raise ValueError(f"Unknown dataset: {dataset}.  "
                             f"Choose from: {list(self.PROFILES.keys())}")
        self.dataset = dataset
        p = self.PROFILES[dataset]

        self.max_turns            = p['max_turns']
        self.initial_sentiment    = p['initial_sentiment']
        self.initial_frustration  = p['initial_frustration']
        self.resolution_threshold = p['resolution_threshold']
        self.info_gain_rate       = p['info_gain_rate']
        self.sentiment_boost      = p['sentiment_boost']
        self.dataset_name         = p['name']

        self.reset()

    # ── Core API ─────────────────────────────────────────────────────────

    def reset(self):
        """Start a new conversation episode."""
        self.sentiment      = float(np.clip(self.initial_sentiment     + np.random.normal(0, 0.10), 0, 1))
        self.frustration    = float(np.clip(self.initial_frustration   + np.random.normal(0, 0.10), 0, 1))
        self.info_gathered  = 0.0
        self.turn           = 0
        self.done           = False
        self.outcome        = None

        # history tracking  (needed for the richer 15-dim state)
        self.last_action         = -1             # -1 = no action yet
        self.strategy_counts     = np.zeros(self.NUM_ACTIONS)
        self.consecutive_count   = 0              # how many times same action in a row

        # conversation transcript  (for the LLM pipeline)
        self.transcript: list[dict] = []
        opening = random.choice(_OPENING_UTTERANCES[self.dataset])
        self.transcript.append({"role": "customer", "text": opening})
        self._last_customer_utterance = opening

        return self._build_state()

    def step(self, action: int):
        """
        Execute one conversation turn.

        Parameters
        ----------
        action : int
            Index 0-4 corresponding to one of the 5 strategies.

        Returns
        -------
        next_state : np.ndarray  (shape: [15])
        reward     : float
        done       : bool
        info       : dict  — includes 'outcome', 'context_for_llm'
        """
        if self.done:
            return self._build_state(), 0.0, True, {"outcome": self.outcome}

        action_name = self.ACTIONS[action]
        prev_sentiment    = self.sentiment
        prev_frustration  = self.frustration
        self.turn        += 1

        # ── update history features ────────────────────────────────────
        if action == self.last_action:
            self.consecutive_count += 1
        else:
            self.consecutive_count  = 1
        self.last_action = action
        self.strategy_counts[action] += 1

        # ── base step cost ─────────────────────────────────────────────
        reward = -0.02

        # ── action dynamics ────────────────────────────────────────────
        if action_name == 'ask_info':
            gain = self.info_gain_rate + np.random.normal(0, 0.05)
            self.info_gathered = min(1.0, self.info_gathered + gain)
            # diminishing returns: too many questions frustrate the customer
            if self.turn > 3 or self.strategy_counts[0] > 2:
                self.frustration = min(1.0, self.frustration + 0.06 * (self.strategy_counts[0] - 2))
                self.sentiment   = max(0.0, self.sentiment - 0.03)
            reward += 0.10

        elif action_name == 'provide_solution':
            if self.info_gathered >= self.resolution_threshold:
                p_success = 0.55 + 0.35 * self.info_gathered
                if np.random.random() < p_success:
                    self.done    = True
                    self.outcome = 'resolved'
                    reward      += 5.0
                    self.sentiment = min(1.0, self.sentiment + 0.30)
                else:
                    self.frustration = min(1.0, self.frustration + 0.12)
                    reward          -= 0.5
            else:
                if np.random.random() < 0.12:
                    self.done    = True
                    self.outcome = 'resolved'
                    reward      += 2.5
                else:
                    self.frustration = min(1.0, self.frustration + 0.22)
                    self.sentiment   = max(0.0, self.sentiment - 0.18)
                    reward          -= 1.2

        elif action_name == 'empathize':
            self.sentiment   = min(1.0, self.sentiment   + self.sentiment_boost)
            self.frustration = max(0.0, self.frustration - 0.10)
            reward          += 0.20
            # repeat empathize is less effective
            if self.strategy_counts[2] > 2:
                reward -= 0.10 * (self.strategy_counts[2] - 2)

        elif action_name == 'escalate':
            self.done    = True
            self.outcome = 'escalated'
            reward      -= 3.0

        elif action_name == 'close':
            self.done = True
            if self.info_gathered >= self.resolution_threshold and self.sentiment > 0.50:
                self.outcome = 'resolved'
                reward      += 2.0
            else:
                self.outcome = 'abandoned'
                reward      -= 2.0

        # ── natural dynamics ───────────────────────────────────────────
        if not self.done and self.turn >= self.max_turns:
            self.done    = True
            self.outcome = 'abandoned'
            reward      -= 2.0

        # frustration erodes sentiment over time
        if self.frustration > 0.7:
            self.sentiment = max(0.0, self.sentiment - 0.04)

        self.sentiment   = float(np.clip(self.sentiment,   0, 1))
        self.frustration = float(np.clip(self.frustration, 0, 1))

        # ── build customer utterance for transcript ────────────────────
        delta_sentiment = self.sentiment - prev_sentiment
        customer_reply  = self._sample_customer_utterance(action_name, delta_sentiment)
        self.transcript.append({"role": "agent_strategy", "text": action_name})
        self.transcript.append({"role": "customer",       "text": customer_reply})
        self._last_customer_utterance = customer_reply

        next_state = self._build_state()
        info = {
            "outcome":         self.outcome,
            "context_for_llm": self.get_llm_context(),
            "delta_sentiment": delta_sentiment,
        }
        return next_state, float(reward), self.done, info

    # ── State construction ────────────────────────────────────────────────

    def _build_state(self) -> np.ndarray:
        """Assemble the 15-dimensional state vector."""
        # [0-3] core signals
        core = np.array([
            self.sentiment,
            self.frustration,
            self.info_gathered,
            self.turn / self.max_turns,
        ])

        # [4-8] last action one-hot  (all zeros on first turn)
        last_oh = np.zeros(self.NUM_ACTIONS)
        if self.last_action >= 0:
            last_oh[self.last_action] = 1.0

        # [9-13] strategy usage frequency  (normalised by total turns)
        total = max(self.turn, 1)
        usage_norm = self.strategy_counts / total

        # [14] repetition signal
        repeat = float(self.consecutive_count >= 3)

        return np.concatenate([core, last_oh, usage_norm, [repeat]])

    # ── LLM context ───────────────────────────────────────────────────────

    def get_llm_context(self) -> dict:
        """Return context dict that strategy_prompts.get_prompt() accepts."""
        return {
            "turn":           self.turn,
            "sentiment":      self.sentiment,
            "frustration":    self.frustration,
            "info_gathered":  self.info_gathered,
            "last_utterance": self._last_customer_utterance,
            "dataset":        self.dataset_name,
        }

    # ── Transcript rendering ──────────────────────────────────────────────

    def render_transcript(self):
        """Pretty-print the conversation transcript so far."""
        print(f"\n{'─' * 60}")
        print(f"  Conversation Transcript  ({self.dataset_name})")
        print(f"{'─' * 60}")
        for i, entry in enumerate(self.transcript):
            role = entry["role"]
            text = entry["text"]
            if role == "customer":
                print(f"  Customer : {text}")
            elif role == "agent_strategy":
                print(f"  [Strategy selected: {text}]")
            elif role == "agent_response":
                print(f"  Agent    : {text}")
        print(f"{'─' * 60}")
        if self.done:
            print(f"  Outcome  : {self.outcome.upper()}")
        print(f"  State    : sentiment={self.sentiment:.2f}  "
              f"frustration={self.frustration:.2f}  "
              f"info={self.info_gathered:.2f}  "
              f"turn={self.turn}/{self.max_turns}")
        print(f"{'─' * 60}\n")

    def add_agent_response_to_transcript(self, response: str):
        """Store the LLM's generated response in the transcript."""
        self.transcript.append({"role": "agent_response", "text": response})

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _sample_customer_utterance(action_name: str, delta_sentiment: float) -> str:
        sentiment_dir = "positive" if delta_sentiment >= 0 else "negative"
        bank = _UTTERANCES.get(action_name, {}).get(sentiment_dir, ["..."])
        return random.choice(bank)

    def get_state(self) -> np.ndarray:
        return self._build_state()

    @staticmethod
    def get_state_dim() -> int:
        return MDPCustomerServiceEnv.STATE_DIM

    @staticmethod
    def get_num_actions() -> int:
        return MDPCustomerServiceEnv.NUM_ACTIONS


# ── Quick smoke-test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    for ds in ['twitter', 'reddit', 'openassistant']:
        env = MDPCustomerServiceEnv(ds)
        state = env.reset()
        print(f"\nDataset: {ds}  |  state dim: {len(state)}")
        while not env.done:
            action = np.random.randint(env.NUM_ACTIONS)
            state, reward, done, info = env.step(action)
            print(f"  turn={env.turn}  action={env.ACTIONS[action]:<20s}  "
                  f"reward={reward:+.2f}  done={done}")
        print(f"  >> Outcome: {env.outcome}")
        env.render_transcript()
