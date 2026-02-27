# Rewards & Escalation

> **TL;DR:** The reward function is a direct translation of economic cost into a number the agent can optimise. Every term has a real-world monetary interpretation.

---

## 1. What Counts as Escalation?

Escalation means **handing a conversation from the automated bot to a human Customer Support Agent (CSA)**. This is a cost event — both direct (CSA time) and indirect (opportunity cost of the CSA's capacity).

### Hard Escalation Triggers (rule-based signals)

The following are used as ground-truth escalation labels in training data:

| Signal | Source | Threshold |
|--------|--------|-----------|
| Customer explicitly requests human / manager | Text match | Any of 18 phrases |
| Conversation went unanswered | Twitter metadata | Company replied 0 times |
| Conversation is very long | Turn count | > 8 turns |
| Sentiment never improved | Sentiment trajectory | Final sentiment < −0.2 AND slope < 0 |
| Low quality score | OpenAssistant labels | Mean quality < 3.0 / 5.0 |

### Escalation Phrase Vocabulary

Phrases that, if detected, trigger the escalation signal:

```
"speak to manager"     "talk to a person"     "human agent"
"not resolved"         "still not working"    "this is unacceptable"
"cancel my subscription" "worst service"      "terrible experience"
"file a complaint"     "escalate"             "supervisor"
"refund"               "compensation"         "legal action"
"been waiting"         "no response"          "ignored"
```

> **Assumption:** This is a proxy label. Real escalation labels would come from CRM system handoff records. In their absence, we use the above heuristics. See [AML Business Notes](aml_business_notes.md#assumption-catalogue) for annotation reliability discussion.

---

## 2. The Economic Reward Function

### Full Equation

$$R_t = \underbrace{+\alpha \cdot \mathbf{1}_{\text{Resolved}}}_{\text{Resolution bonus}}\ -\ \underbrace{C_{\text{esc}} \cdot \mathbf{1}_{\text{Escalate}}}_{\text{Escalation cost}}\ -\ \underbrace{L_{\text{churn}} \cdot \mathbf{1}_{\text{Failure}}}_{\text{Churn loss}}\ -\ \underbrace{\delta \cdot T}_{\text{Turn penalty}}$$

### Term-by-Term Breakdown

| Term | Symbol | Value | What it captures |
|------|--------|-------|-----------------|
| Resolution bonus | `α = 5.0` | +5.0 | Bot successfully resolved issue; no human needed |
| Escalation cost | `C_esc` | Tier-specific (below) | Direct cost of CSA time + overhead |
| Churn loss | `L_churn = churn_prob × CLV / 1000` | Tier-scaled | Probability-weighted lifetime value lost if issue fails |
| Turn penalty | `δ = 0.1 per turn` | −0.1 × T | Longer conversations consume more resources |

### Tier-Specific Escalation Costs

| Tier | `C_esc` (in reward units) | Real-world cost proxy |
|------|--------------------------|----------------------|
| Free | 0 | No human support allocated |
| Pro | −0.30 | ₹15 CSA time / 50 (normalised) |
| Business+ | −1.00 | ₹50 CSA time / 50 |
| Enterprise | −2.00 | ₹100 CSA time / 50 |

### Satisfaction Component

An additional `+sat_score × 2 − 1` term is included based on estimated customer sentiment at episode end:

- Perfect sentiment (+1.0) adds +1.0 to the reward
- Neutral (0.5) adds 0
- Negative (0.0) subtracts −1.0

---

## 3. The Economically Optimal Escalation Threshold

This is the **central insight** of the AML proposal. Rather than training the model to "predict escalation", we derive the exact failure probability above which escalation saves money.

### The Decision Rule

> Escalate if:  `p > τ_tier`

where `p` is the estimated probability that the bot will fail, and `τ_tier` is:

$$\tau_{\text{tier}} = \frac{C_{\text{escalation}}}{\text{churn\_prob} \times \text{CLV}}$$

### Computed Thresholds

| Tier | Escalation Cost | Churn Prob | CLV | Threshold τ |
|------|----------------|-----------|-----|-------------|
| Free | ₹0 | 15% | ₹0 | **1.00** (never escalate — no cost savings possible) |
| Pro | ₹15 | 8% | ₹4,032 | **0.047** (escalate if > 4.7% failure probability) |
| Business+ | ₹50 | 3% | ₹12,852 | **0.130** (escalate if > 13% failure probability) |
| Enterprise | ₹100 | 2% | ₹28,800 | **0.174** (escalate if > 17.4% failure probability) |

### Intuition

- **Free tier threshold = 1.0:** The CLV is zero, so there's no expected churn cost to justify spending a CSA's time. Route everything to the bot.
- **Pro threshold = 4.7%:** Pro customers have moderate CLV (₹4,032). Even a 5% risk of failure is worth escalating because the expected churn loss (₹322) exceeds the escalation cost (₹15).
- **Enterprise threshold = 17.4%:** Enterprise has very high CLV (₹28,800) but also higher escalation cost (₹100). The threshold is actually *higher* than Pro because the ratio is less extreme — but once the probability exceeds ~17%, the expected churn loss (~₹576 per conversation at risk) outweighs the escalation cost.

---

## 4. Example Reward Calculation

**Scenario:** Pro customer, billing issue, 3 turns, bot successfully resolved.

```
Resolution bonus:    +5.0               (issue resolved)
Escalation cost:      0.0               (bot handled it — cost saved = +0.30)
  → cost-saving bonus: +0.30
Churn loss:           0.0               (resolved → no churn)
Turn penalty:        −0.1 × 3 = −0.30  (3 turns)
Satisfaction:        +0.5               (positive final sentiment)
Inference cost:      −0.08              (₹0.80 / 10 normalised)

Total reward:  +5.0 + 0.30 + 0 − 0.30 + 0.5 − 0.08 = +5.42
```

**Scenario:** Enterprise customer, outage complaint, 8 turns, bot failed, no escalation.

```
Resolution bonus:     0.0               (not resolved)
Churn loss:          −(0.02 × 28800)/1000 = −0.576  (expected CLV loss)
Escalation cost:      0.0               (didn't escalate — no cost saved either)
Turn penalty:        −0.1 × 8 = −0.80  (8 turns)
Satisfaction:        −1.0               (very negative final sentiment)
Inference cost:      −0.08

Total reward:  0 − 0.576 − 0 − 0.80 − 1.0 − 0.08 = −2.456
```

The contrast illustrates the design intent: the agent is strongly incentivised to either resolve or escalate early for Enterprise customers, not to keep trying with the bot on complex issues.

---

## 5. Capacity Constraint & Priority Queue

When multiple conversations need escalation simultaneously, the system can't always comply — there are only `K = 50` human slots per hour.

**Priority score formula:**

$$\text{Priority} = p_{\text{escalation}} \times (\text{CLV} - C_{\text{interaction}}) \times \left(1 - \frac{\text{CSAT}}{5}\right)$$

Where:
- `p_escalation` = estimated failure probability (from context features)
- `CLV − C_interaction` = net value at stake
- `(1 − CSAT/5)` = urgency: lower current satisfaction → higher priority

Conversations are ranked descending and the top-K get human agents. The rest fall back to the bot.

---

## 6. Reward Model (Learned from OpenAssistant)

In addition to the hand-crafted economic reward, we train a secondary **learned reward model** on OpenAssistant quality ratings. This gives us a data-driven signal for conversation quality.

**Architecture:** Ridge regression on the 23-dim feature vector.

**Training data:** Labelled subset of OpenAssistant conversations where:
- Each conversation has a `quality_score` ∈ [1.0, 5.0]
- Normalised to [−1, +1]

**Usage:** At inference time, the reward model can *supplement* the economic reward as a CSAT proxy. This is particularly useful for tiers where economic data is sparse (e.g., Free tier with CLV = 0).

```
shaped_reward = 0.7 × economic_reward + 0.3 × learned_quality_reward
```

---

*Back: [MDP Formulations](mdp_formulation.md) | Next: [AML Business Notes](aml_business_notes.md)*
