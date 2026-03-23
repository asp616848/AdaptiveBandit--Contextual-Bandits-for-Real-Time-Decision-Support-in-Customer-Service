import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import os
import json

# ACTION SPACE:
# 0: Ask_for_Information
# 1: Provide_Solution
# 2: Affective_Repair
# 3: Escalate_to_Human
# 4: Close_with_Feedback
# 5: Proactive_Update
# 6: Set_Expectation

ACTIONS = ["Ask_for_Information", "Provide_Solution", "Affective_Repair", 
           "Escalate_to_Human", "Close_with_Feedback", "Proactive_Update", "Set_Expectation"]

# SENTIMENT SPACE:
# 0: Angry, 1: Neutral, 2: Happy
SENTIMENTS = ["Angry", "Neutral", "Happy"]

class CustomerSupportEnv(gym.Env):
    """
    A custom Gymnasium environment representing a Customer Support Dialogue.
    Dynamically loads statistical probabilities from real-world datasets!
    """
    def __init__(self, max_turns=10, matrix_path="empirical_transition_matrix_twitter.json"):
        super(CustomerSupportEnv, self).__init__()
        
        self.max_turns = max_turns
        
        # Action space: 7 discrete actions
        self.action_space = spaces.Discrete(7)
        
        # State space: [Sentiment (0-2), Frustration (0-5), Turn Count (0-max_turns), Resolution Flag (0 or 1)]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([2, 5, self.max_turns, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        self.state = None
        
        # Locate the matrix regardless of cwd
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, matrix_path)
        
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                self.matrix = json.load(f)
            self.use_empirical = True
        else:
            self.use_empirical = False
            print(f"Warning: Empirical Matrix not found at {full_path}. Falling back to synthetic probabilities.")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initial state distribution: Customers usually start Angry (0) or Neutral (1)
        initial_sentiment = np.random.choice([0, 1], p=[0.7, 0.3])
        self.state = np.array([initial_sentiment, 0.0, 0.0, 0.0], dtype=np.float32)
        
        return self.state, {}

    def step(self, action):
        sentiment, frustration, turn, resolved = self.state
        turn += 1
        
        # --- EMPIRICAL TRANSITION RULES ---
        if self.use_empirical:
            state_key = str(int(sentiment))
            action_key = str(int(action))
            
            if state_key in self.matrix and action_key in self.matrix[state_key]:
                probs = self.matrix[state_key][action_key]
                probs = np.array(probs) / np.sum(probs) # Normalize
                next_sentiment = np.random.choice([0, 1, 2], p=probs)
                
                sentiment = float(next_sentiment)
                # Logical flags for episode termination based on sentiment shift
                if action == 1 and sentiment == 2:
                    resolved = 1.0 # Provided solution and they are happy -> resolved
                elif action == 1 and sentiment == 0:
                    frustration = min(5.0, frustration + 1.0) # Failed solution makes them mad
                elif action == 2:
                    frustration = max(0.0, frustration - 1.0) # Apology reduces frustration state
                elif action == 3:
                    resolved = 1.0 # Escalated to human (ended)
                elif action == 4 and resolved == 0.0:
                    frustration = 5.0 # Closed without fixing
            else:
                sentiment = np.random.choice([0, 1, 2])
        else:
            # --- SYNTHETIC TRANSITION RULES ---
            if action == 1:
                if sentiment == 0:
                    success = np.random.rand() < 0.3
                else:
                    success = np.random.rand() < 0.7
                if success:
                    resolved = 1.0
                    sentiment = 2.0
                else:
                    frustration = min(5.0, frustration + 1.0)
                    sentiment = 0.0
            elif action == 2:
                if sentiment == 0:
                    sentiment = np.random.choice([0.0, 1.0], p=[0.2, 0.8])
                frustration = max(0.0, frustration - 1.0)
            elif action == 0:
                if sentiment == 0 and frustration > 2:
                    frustration = min(5.0, frustration + 1.5)
                else:
                    sentiment = 1.0
            elif action == 3:
                resolved = 1.0
            elif action == 4:
                if resolved == 0.0:
                    sentiment = 0.0
                    frustration = 5.0
            else:
                if sentiment == 0:
                    sentiment = 1.0

        # --- REWARD FUNCTION ---
        reward = 0.0
        reward -= 1.0 # Turn penalty
        
        if resolved == 1.0:
            if action == 3:
                reward -= 20.0 # Escalation penalty
            elif action == 1:
                reward += 30.0 # Resolution bonus
                
        terminated = False
        if resolved == 1.0:
            terminated = True
        elif turn >= self.max_turns:
            terminated = True
            reward -= 15.0 # Abandon penalty
            
        if frustration >= 4.0:
            reward -= 5.0 # Frustration penalty

        self.state = np.array([sentiment, frustration, turn, resolved], dtype=np.float32)
        
        return self.state, reward, terminated, False, {}

if __name__ == "__main__":
    env = CustomerSupportEnv()
    obs, _ = env.reset()
    print("Environment initialized correctly.")
    print("Using Empirical Matrix:", getattr(env, 'use_empirical', False))
