"""
Customer Service Conversation Simulator
========================================
Adapted from the RL-Chatbot's UserSimulator + StateTracker pattern.
Instead of restaurant slots, we simulate customer support conversations
with sentiment, frustration, and resolution dynamics.

Each dataset has different conversation characteristics:
- Twitter: short, urgent, customers start frustrated
- Reddit: longer threads, more neutral, slower info gathering
- OpenAssistant: moderate length, cooperative users, quality-focused
"""

import numpy as np
import random

class CustomerServiceEnv:
    
    # the 5 actions our agent can take
    ACTIONS = ['ask_info', 'provide_solution', 'empathize', 'escalate', 'close']
    NUM_ACTIONS = 5
    
    # different profiles based on dataset characteristics from our EDA
    PROFILES = {
        'twitter': {
            'max_turns': 10,           # twitter convos are short
            'initial_sentiment': 0.3,   # customers start unhappy (complaints)
            'initial_frustration': 0.6, # high urgency
            'resolution_threshold': 0.7,
            'info_gain_rate': 0.25,     # info gathered per ask
            'sentiment_boost': 0.15,    # how much empathy helps
            'name': 'Twitter Customer Support'
        },
        'reddit': {
            'max_turns': 20,            # reddit threads are longer
            'initial_sentiment': 0.5,   # more neutral tone
            'initial_frustration': 0.3, # less urgent
            'resolution_threshold': 0.6,
            'info_gain_rate': 0.15,     # slower info gathering
            'sentiment_boost': 0.12,
            'name': 'Reddit DSTC8 Corpus'
        },
        'openassistant': {
            'max_turns': 15,            # moderate length
            'initial_sentiment': 0.5,   # neutral
            'initial_frustration': 0.2, # cooperative users
            'resolution_threshold': 0.5,# easier to resolve
            'info_gain_rate': 0.2,
            'sentiment_boost': 0.18,    # quality interactions matter more
            'name': 'OpenAssistant Conversations'
        }
    }
    
    def __init__(self, dataset='twitter'):
        """set up the environment for a specific dataset"""
        if dataset not in self.PROFILES:
            raise ValueError(f"Unknown dataset: {dataset}. Choose from: {list(self.PROFILES.keys())}")
        
        profile = self.PROFILES[dataset]
        self.max_turns = profile['max_turns']
        self.initial_sentiment = profile['initial_sentiment']
        self.initial_frustration = profile['initial_frustration']
        self.resolution_threshold = profile['resolution_threshold']
        self.info_gain_rate = profile['info_gain_rate']
        self.sentiment_boost = profile['sentiment_boost']
        self.dataset_name = profile['name']
        
        self.reset()
    
    def reset(self):
        """reset the environment for a new episode (new customer conversation)"""
        # add some randomness so every conversation is slightly different
        self.sentiment = np.clip(self.initial_sentiment + np.random.normal(0, 0.1), 0, 1)
        self.frustration = np.clip(self.initial_frustration + np.random.normal(0, 0.1), 0, 1)
        self.info_gathered = 0.0
        self.turn = 0
        self.done = False
        self.outcome = None  # will be 'resolved', 'escalated', or 'abandoned'
        return self.get_state()
    
    def get_state(self):
        """return current state as a numpy array"""
        return np.array([
            self.sentiment,
            self.frustration,
            self.info_gathered,
            self.turn / self.max_turns  # normalized turn number
        ])
    
    def get_state_size(self):
        return 4
    
    def step(self, action):
        """
        take an action and return (next_state, reward, done, outcome)
        this is where the conversation dynamics happen
        """
        if self.done:
            return self.get_state(), 0, True, self.outcome
        
        action_name = self.ACTIONS[action]
        self.turn += 1
        reward = -0.02  # small penalty per turn to encourage efficiency
        
        # --- action effects ---
        
        if action_name == 'ask_info':
            # we gather information from the customer
            info_gain = self.info_gain_rate + np.random.normal(0, 0.05)
            self.info_gathered = min(1.0, self.info_gathered + info_gain)
            
            # asking too many questions gets annoying
            if self.turn > 3:
                self.frustration = min(1.0, self.frustration + 0.05)
                self.sentiment = max(0, self.sentiment - 0.03)
            reward += 0.1
        
        elif action_name == 'provide_solution':
            if self.info_gathered >= self.resolution_threshold:
                # we have enough info, good chance of solving it
                success_prob = 0.6 + 0.3 * self.info_gathered
                if np.random.random() < success_prob:
                    self.done = True
                    self.outcome = 'resolved'
                    reward += 5.0
                    self.sentiment = min(1.0, self.sentiment + 0.3)
                else:
                    # solution didn't work
                    self.frustration = min(1.0, self.frustration + 0.1)
                    reward -= 0.5
            else:
                # trying to solve without enough info - risky
                if np.random.random() < 0.15:
                    # lucky guess
                    self.done = True
                    self.outcome = 'resolved'
                    reward += 3.0
                else:
                    # bad solution makes customer more frustrated
                    self.frustration = min(1.0, self.frustration + 0.2)
                    self.sentiment = max(0, self.sentiment - 0.15)
                    reward -= 1.0
        
        elif action_name == 'empathize':
            # showing empathy improves sentiment and reduces frustration
            self.sentiment = min(1.0, self.sentiment + self.sentiment_boost)
            self.frustration = max(0, self.frustration - 0.1)
            reward += 0.2
        
        elif action_name == 'escalate':
            # hand off to human agent - we lose
            self.done = True
            self.outcome = 'escalated'
            reward -= 3.0
        
        elif action_name == 'close':
            self.done = True
            if self.info_gathered >= self.resolution_threshold and self.sentiment > 0.5:
                self.outcome = 'resolved'
                reward += 2.0
            else:
                self.outcome = 'abandoned'
                reward -= 2.0
        
        # --- natural dynamics ---
        
        # max turns reached
        if not self.done and self.turn >= self.max_turns:
            self.done = True
            self.outcome = 'abandoned'
            reward -= 2.0
        
        # high frustration naturally decreases sentiment
        if self.frustration > 0.7:
            self.sentiment = max(0, self.sentiment - 0.05)
        
        # keep values in bounds
        self.sentiment = np.clip(self.sentiment, 0, 1)
        self.frustration = np.clip(self.frustration, 0, 1)
        
        return self.get_state(), reward, self.done, self.outcome
    
    def render(self, action=None):
        """print current state nicely"""
        action_str = f" | Action: {self.ACTIONS[action]}" if action is not None else ""
        print(f"  Turn {self.turn:2d}{action_str}")
        print(f"    Sentiment: {self.sentiment:.2f} | Frustration: {self.frustration:.2f} | Info: {self.info_gathered:.2f}")
        if self.done:
            print(f"    >> Outcome: {self.outcome}")
