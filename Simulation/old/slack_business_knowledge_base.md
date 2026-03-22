# Slack Business Context & Cost Knowledge Base

## 1. Company Profile
- **Domain:** Tiered B2B SaaS platform (Slack-like architecture and pricing).
- **Core Problem:** Escalating support tickets to human agents is highly expensive, but failing to successfully support high-value users leads to disproportionate churn losses.

## 2. Tier Economics
We utilize 4 primary tiers with the following monthly economic profiles:
- **Free:** 
  - Customer Value: -Rs.20 (Network effect acquisition asset)
  - Escalation Strategy: Immediate deflection or pure AI-bot. Minimal human escalation allowed.
- **Pro:** 
  - Customer Value: Rs.168 Profit (Margin 68%). 
  - Base Churn Risk: ~8% if bad experience. Escalation Cost: Rs.15 per ticket.
- **Business+:** 
  - Customer Value: Rs.357 Profit (Margin 64%).
  - Base Churn Risk: ~3% if bad experience, but heavy absolute dollar penalty. Escalation Cost: Rs.50 per ticket.
- **Enterprise+:** 
  - Customer Value: Massive Life-Time Value (CLV).
  - Escalation Strategy: Retention critical. Default to human escalation if initial bot triage fails or frustration spikes. 

## 3. Simulator Assumptions
- Automated bot cost is negligible compared to a human intervention (ratio 1:10).
- **Global CSAT Constraint:** Maintain a minimum average CSAT floor of 3.5/5.0 across the user base.
- An optimal Contextual Bandit/RL model should learn to prioritize human handoffs for frustrated Business+/Enterprise users and aggressively use fallback/resolution loops for Free/Pro tiers.

## 4. Simulator Environment Rules
The state variables of the simulation reflect:
- Past actions implemented in the current conversation thread.
- Emotional proxies (`sentiment_score`, `frustration_proxy`).
- Resolution probability and Escalation likelihood. 
When prompting LLMs for synthetic generation, enforce these Slack parameters so the agent respects tier limitations.