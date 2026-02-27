# AdaptiveBandit — System Overview

> **One-line summary:** Train a system that decides, in real time, whether to let a chatbot handle a customer support conversation or hand it to a human agent — optimising for profit, not just accuracy.

---

## The Core Business Problem

Every time a customer contacts support, the company faces a routing decision:

```
Incoming conversation
        │
        ▼
┌───────────────────┐       ┌──────────────────────┐
│   Route to BOT    │  OR   │  Route to HUMAN agent │
│  - Cheap (~₹0.8)  │       │  - Expensive (~₹33)  │
│  - Fast           │       │  - Slower             │
│  - Risk: failure  │       │  - Risk: over-use     │
└───────────────────┘       └──────────────────────┘
```

Getting this wrong in either direction costs money:

| Mistake | What happened | Cost |
|---------|--------------|------|
| **Under-escalation** | Bot failed; customer churned | Lost CLV (₹4,032–₹28,800) |
| **Over-escalation** | Sent humans work the bot could handle | ₹33 wasted per conversation |

The goal is to find the **optimal threshold** — specific to each customer tier — that minimises total expected cost.

---

## The Two-Phase Architecture

### Phase I — Contextual Bandit (one routing decision per conversation)

> *"Should this conversation go to a bot or human, given what I know right now?"*

- Single binary decision at the start of the conversation
- Works like a smart A/B test that updates its own weights
- Appropriate for **Free** and **Pro** tier (low stakes, high volume)

### Phase II — Full RL / DQN (multi-turn dialogue planning)

> *"What is the best sequence of actions across the entire conversation?"*

- Sequence of decisions: ask clarifying questions → try solution → escalate or close
- More powerful but computationally heavier
- Appropriate for **Business+** and **Enterprise** (high CLV, complex issues)

---

## Four Agents Compared

| Agent | Strategy | Actions | Phase |
|-------|----------|---------|-------|
| **Rule-Based** | Hard-coded: Free always bot, Enterprise always human | Binary | Baseline |
| **LinUCB** | Learn linear reward model per action; explore via UCB | Binary | I |
| **Thompson Sampling** | Bayesian posterior — sample, act, update beliefs | Binary | I |
| **DQN** | Neural Q-network; learn end-to-end dialogue policy | 4-way | II |

---

## The Tier System

Pricing is modelled after Slack's tier structure (INR per user per month):

| Tier | Monthly Profit | Churn Risk | CLV | Business Driver |
|------|---------------|-----------|-----|-----------------|
| **Free** | −₹20 | 15% | ₹0 | Network effect (upsell pipeline) |
| **Pro** | +₹168 | 8% | ₹4,032 | Volume cash cow |
| **Business+** | +₹357 | 3% | ₹12,852 | Retention-critical |
| **Enterprise** | +₹600 | 2% | ₹28,800 | Retention-critical, strategic |

> **Why does tier matter?** The cost of a churn event scales with CLV. Losing an Enterprise customer is 7× worse than losing a Business+ customer. So the system should be far more aggressive about escalating Enterprise conversations.

---

## Files at a Glance

```
CustomerSupportBandit/
├── config.py                  ← All economic parameters — edit here first
├── train.py                   ← Training loops for all agents
├── run_experiment.py          ← CLI entry point
├── gemini_utils.py            ← Optional LLM-as-judge enrichment
│
├── data/
│   ├── pipeline.py            ← Loads + merges all datasets
│   ├── twitter_loader.py      ← Twitter CS conversation threads
│   ├── openassistant_loader.py← OA conversation trees + quality ratings
│   └── feature_engineer.py   ← Converts raw text → 23-dim feature vector
│
├── agents/
│   ├── rule_based.py          ← Deterministic baseline
│   ├── linucb.py              ← LinUCB contextual bandit
│   ├── thompson_sampling.py   ← Bayesian TS contextual bandit
│   └── dqn_agent.py           ← DQN with replay buffer
│
├── environment/
│   └── customer_env.py        ← Gym-like environment
│
├── rewards/
│   ├── economic_reward.py     ← Tier-specific reward + escalation thresholds
│   └── reward_model.py        ← Learned reward from OA quality ratings
│
└── evaluation/
    ├── metrics.py             ← Business metrics (cost saved, CSAT proxy)
    ├── sensitivity.py         ← CSA cost / churn elasticity sweeps
    └── visualize.py           ← All plots
```

---

## Quick-Start

```bash
# Synthetic data, 3000 episodes, no plots
python run_experiment.py --episodes 3000 --no-plots

# Real Twitter + OA data, with Gemini enrichment
python run_experiment.py --real-data --gemini --episodes 5000

# Notebook (interactive)
jupyter notebook main_experiment.ipynb
```

---

*Next: [Data & Simulation](data.md) | [MDP Formulations](mdp_formulation.md) | [Rewards & Escalation](reward_and_escalation.md) | [AML Business Notes](aml_business_notes.md)*
