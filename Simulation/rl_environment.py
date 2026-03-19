import numpy as np
import random

# Core canonical action space derived from Part 2 specifications
ACTION_SPACE = [
    "Ask_for_Information",
    "Provide_Solution",
    "Affective_Repair",
    "Escalate_to_Human",
    "Close_with_Feedback",
    "Proactive_Update",
    "Set_Expectation",
]

# Business and Tier Economics (Derived from AML & RL Proposals)
TIER_ECONOMICS = {
    "Free": {"escalation_cost": 0, "churn_risk_base": 0.0, "profit": -20},
    "Pro": {"escalation_cost": 15, "churn_risk_base": 0.08, "profit": 168},
    "Business+": {"escalation_cost": 50, "churn_risk_base": 0.03, "profit": 357},
    "Enterprise": {"escalation_cost": 100, "churn_risk_base": 0.01, "profit": 1000} # Estimated high value CLI
}

PERSONAS = ["Cooperative", "Impatient", "Escalation-prone", "Silent-dropoff"]

class CustomerSupportEnv:
    """
    RL-focused Simulator tracking state variables, past actions, and tier-specific context.
    Forms the backbone matrix for RAG-LLM injection and contextual bandit training.
    """
    def __init__(self, use_llm_rag=False):
        self.action_space = ACTION_SPACE
        self.state = {}
        self.max_turns = 10
        self.use_llm_rag = use_llm_rag
        
    def reset(self, tier=None, persona=None):
        self.current_tier = tier if tier else np.random.choice(list(TIER_ECONOMICS.keys()), p=[0.4, 0.3, 0.2, 0.1])
        self.current_persona = persona if persona else np.random.choice(PERSONAS)
        
        # Base emotional state dependent on persona clustering outputs
        start_sentiment = -0.5 if self.current_persona in ["Impatient", "Escalation-prone"] else 0.0
        start_frustration = 0.6 if self.current_persona == "Impatient" else 0.2
        
        self.state = {
            "tier": self.current_tier,
            "persona": self.current_persona,
            "turn_count": 0,
            "sentiment_score": start_sentiment, # -1.0 to 1.0
            "sentiment_delta": 0.0,
            "frustration_proxy": start_frustration, # 0.0 to 1.0 
            "escalation_likelihood": 0.5 if self.current_persona == "Escalation-prone" else 0.1,
            "resolution_probability": 0.0,
            "past_actions": []  # List representing dialogue history
        }
        return self.get_observation()
        
    def get_observation(self):
        """Returns the fully observable state for the RL Agent"""
        return self.state.copy()
        
    def step(self, action_idx):
        action = self.action_space[action_idx]
        
        old_sentiment = self.state["sentiment_score"]
        
        # Add to historical context
        self.state["past_actions"].append(action)
        self.state["turn_count"] += 1
        
        # Determine internal transition dynamics mapped from Proposal Realism
        if action == "Provide_Solution":
            self.state["resolution_probability"] += np.random.uniform(0.1, 0.5)
            self.state["sentiment_score"] += 0.2
        elif action == "Affective_Repair":
            self.state["frustration_proxy"] = max(0.0, self.state["frustration_proxy"] - 0.3)
            self.state["sentiment_score"] += 0.1
        elif action == "Escalate_to_Human":
            self.state["escalation_likelihood"] = 1.0 # Trigger event
            self.state["resolution_probability"] += 0.8 # Human proxy fixes it mostly
        elif action == "Ask_for_Information":
            if self.state["persona"] == "Impatient":
                self.state["frustration_proxy"] += 0.3
                self.state["sentiment_score"] -= 0.3
            else:
                self.state["resolution_probability"] += 0.15
        elif action == "Set_Expectation":
            self.state["frustration_proxy"] *= 0.8
            self.state["escalation_likelihood"] -= 0.1
                
        # Clip state bounds to sensible statistical constraints
        self.state["sentiment_score"] = np.clip(self.state["sentiment_score"], -1.0, 1.0)
        self.state["frustration_proxy"] = np.clip(self.state["frustration_proxy"], 0.0, 1.0)
        self.state["escalation_likelihood"] = np.clip(self.state["escalation_likelihood"], 0.0, 1.0)
        self.state["resolution_probability"] = np.clip(self.state["resolution_probability"], 0.0, 1.0)
        
        self.state["sentiment_delta"] = self.state["sentiment_score"] - old_sentiment
        
        # Terminal condition evaluation
        is_terminal = False
        outcome = "ongoing"
        
        if self.state["resolution_probability"] >= 0.8:
            is_terminal = True
            outcome = "resolved"
        elif self.state["escalation_likelihood"] >= 0.9 and action == "Escalate_to_Human":
            is_terminal = True
            outcome = "escalated"
        elif self.state["frustration_proxy"] >= 0.9 and self.state["persona"] == "Silent-dropoff":
            is_terminal = True
            outcome = "abandoned"
        elif self.state["turn_count"] >= self.max_turns:
            is_terminal = True
            outcome = "abandoned"
            
        reward = self._compute_reward(outcome, action)
        
        return self.get_observation(), reward, is_terminal, {"outcome": outcome}
        
    def _compute_reward(self, outcome, action):
        """Calculates Shaped Reward aligned with Business SaaS Assumptions"""
        tier_data = TIER_ECONOMICS[self.state["tier"]]
        reward = 0.0
        
        # Baseline conversational friction penalty
        reward -= 0.5 
        
        # Action-associated costs
        if action == "Escalate_to_Human":
            reward -= (tier_data["escalation_cost"] * 0.1) 
        else:
            reward -= 0.1 # Automated processing cost proxy
            
        # Terminal Outcome Rewards/Penalties
        if outcome == "resolved":
            reward += 10.0  
            if tier_data["profit"] > 0:
                reward += (tier_data["profit"] * 0.05) 
        elif outcome == "abandoned":
            reward -= 10.0
            reward -= (tier_data["profit"] * tier_data["churn_risk_base"])
            
        # Ongoing engagement bonus/penalty for sentiment delta
        reward += (self.state["sentiment_delta"] * 2.0)
        
        return reward
