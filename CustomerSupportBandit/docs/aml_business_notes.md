# AML for Business — Assumptions, Effect Calculations & Deployment Notes

> This document is the **Applied Machine Learning for Business** companion. It covers every assumption made, how the business effect is calculated, where uncertainty is high, and what a production deployment would need.

---

## 1. Assumption Catalogue

Every assumption is tagged with a **sensitivity level**:
- 🔴 **High** — small changes significantly affect results
- 🟡 **Medium** — moderate impact; worth validating
- 🟢 **Low** — results are robust to reasonable variation

---

### 1.1 Economic Assumptions

| # | Assumption | Value used | Sensitivity | Source |
|---|-----------|-----------|-------------|--------|
| A1 | Free tier monthly profit | −₹20/user | 🟢 | Slack pricing analogy |
| A2 | Pro tier monthly profit | +₹168/user | 🟡 | Slack pricing analogy |
| A3 | Business+ monthly profit | +₹357/user | 🟡 | Slack pricing analogy |
| A4 | Enterprise monthly profit | +₹600/user | 🟡 | Slack pricing analogy |
| A5 | CSA hourly cost | ₹250/hr | 🔴 | Indian BPO cost benchmark |
| A6 | Average handle time (AHT) | 8 minutes | 🔴 | Industry average CS ticket |
| A7 | Cost per escalated conversation | ₹33.3 (A5 × A6/60) | 🔴 | Derived from A5 × A6 |
| A8 | Bot inference cost | ₹0.80/conversation | 🟢 | Cloud API pricing estimate |
| A9 | Full RL inference cost | ₹2.50/conversation | 🟡 | Includes LLM calls if Gemini used |

---

### 1.2 Churn & Retention Assumptions

| # | Assumption | Value | Sensitivity | Notes |
|---|-----------|-------|-------------|-------|
| B1 | Free tier churn after bad experience | 15% | 🟡 | Lower CLV limits impact |
| B2 | Pro tier churn after bad experience | 8% | 🔴 | Main revenue segment |
| B3 | Business+ churn after bad experience | 3% | 🔴 | High CLV makes this expensive |
| B4 | Enterprise churn after bad experience | 2% | 🔴 | Contractual; longer lead time |
| B5 | Pro CLV = 24 months × ₹168 | ₹4,032 | 🟡 | Average tenure assumed |
| B6 | Business+ CLV = 36 months × ₹357 | ₹12,852 | 🔴 | Longer tenure assumed |
| B7 | Enterprise CLV = 48 months × ₹600 | ₹28,800 | 🔴 | Highest uncertainty |
| B8 | "Bad experience" = bot failure to resolve | Binary label | 🔴 | Key proxy assumption |

> **On B8:** We assume that a conversation where the bot failed *and* no escalation occurred equals a "bad experience" that triggers the churn probability. This is a simplification — in reality, customers may not churn even after bad experiences, or may churn despite successful resolutions. A customer survey or CRM churn data would improve this.

---

### 1.3 Data & Label Quality Assumptions

| # | Assumption | Details | Sensitivity |
|---|-----------|---------|-------------|
| C1 | Twitter escalation proxy is valid | Unanswered tweets + phrase detection ≈ 80% precision | 🔴 |
| C2 | OA quality ratings generalise to CS | OA is assistant-style, not support-style | 🟡 |
| C3 | Synthetic data distributions are realistic | Escalation base rates calibrated manually | 🟡 |
| C4 | Tier assignment from Twitter is accurate | Topic-based heuristic, not actual user data | 🔴 |
| C5 | Sentiment scoring is a good failure proxy | VADER/rule-based; validated against OA ratings | 🟡 |

---

### 1.4 Model Assumptions

| # | Assumption | Details | Sensitivity |
|---|-----------|---------|-------------|
| D1 | Reward is linear in features (LinUCB) | Holds if interaction effects are small | 🟡 |
| D2 | IID conversations (bandit) | Conversations are drawn independently | 🟢 |
| D3 | Stationary reward distribution | Customer behaviour doesn't shift over time | 🔴 |
| D4 | 23 features capture sufficient context | No important confounders missing | 🟡 |
| D5 | DQN finds near-optimal policy | May converge to local optima in practice | 🟡 |
| D6 | Episode length ≤ 20 turns | Longer conversations get truncated | 🟢 |

---

## 2. Business Effect Calculation

### 2.1 The Core Formula

The business case rests on one equation:

```
Monthly Net Benefit = Revenue Retained − Cost of Misdirections − System Running Cost
```

Expanded:

```
Monthly Net Benefit
  = (Δ churn_rate × churned_users × avg_CLV_monthly)    ← Retained revenue
  - (n_unnecessary_escalations × cost_per_escalation)   ← Over-escalation waste
  - (n_missed_escalations × churn_prob × avg_CLV)       ← Under-escalation loss
  - (n_conversations × inference_cost)                  ← System cost
```

### 2.2 Worked Example: 10,000 Monthly Conversations

Tier mix: Free 60% | Pro 25% | Business+ 12% | Enterprise 3%

**Baseline (Rule-Based):** Routes all Free to bot, all Enterprise to human, others halfway.

**Improved (LinUCB):** Applies learned thresholds to each conversation individually.

| Tier | Volume | Baseline Churn Events | LinUCB Churn Events | Savings |
|------|--------|----------------------|---------------------|---------|
| Free | 6,000 | 72 (12% × 6000 × 10%) | 60 (learned) | ₹0 (CLV=0) |
| Pro | 2,500 | 50 (2% × 2500) | 35 (improved routing) | 15 × ₹168 = **₹2,520** |
| Business+ | 1,200 | 24 (2% × 1200) | 14 (improved routing) | 10 × ₹357 = **₹3,570** |
| Enterprise | 300 | 3 (1% × 300) | 2 | 1 × ₹600 = **₹600** |

- **Total retained revenue:** ~₹6,690/month
- **Escalation cost reduction** (fewer unnecessary escalations at ₹33 each): ~₹1,500/month
- **System cost** (₹0.80 × 10,000 = ₹8,000/month)
- **Net benefit:** ₹6,690 + ₹1,500 − ₹8,000 = **+₹190/month** at 10K conversations

> **Note:** The system becomes strongly positive at ~50,000 monthly conversations where the retained CLV far exceeds the inference cost.

### 2.3 Sensitivity: CSA Cost (Assumption A5)

We hold all else constant and vary the CSA hourly cost from ₹150 to ₹400/hr:

| CSA Cost/hr | Cost per Escalation | System ROI |
|------------|---------------------|-----------|
| ₹150 | ₹20 | Marginal positive |
| ₹250 (base) | ₹33 | Positive at 50K+ users |
| ₹300 | ₹40 | Positive at 30K+ users |
| ₹400 | ₹53 | Positive at 20K+ users |

**Finding:** ROI is sensitive to A5. If actual CSA costs are lower (e.g., in-house team vs. outsourced), the break-even point shifts significantly.

### 2.4 Sensitivity: Churn Elasticity (Assumption B2–B4)

The most uncertain assumption is whether a bot failure actually causes churn. We parameterise this as `churn_elasticity ∈ [0.5, 2.0]` (multiplier on base churn rates):

| Elasticity | Pro churn/bad exp | Enterprise churn/bad exp | Net Benefit (monthly, 50K convs) |
|-----------|------------------|--------------------------|----------------------------------|
| 0.5 | 4% | 1% | ₹8,000 |
| 1.0 (base) | 8% | 2% | ₹22,000 |
| 1.5 | 12% | 3% | ₹38,000 |
| 2.0 | 16% | 4% | ₹55,000 |

**Finding:** This is the highest-sensitivity assumption. The business case is essentially a bet on whether customers actually churn after bad support experiences. This should be validated with **A/B test data or CRM churn analysis**.

---

## 3. Go / No-Go Framework

The deployment decision uses a structured rule:

```
Deploy if ALL of:
  1. Annual savings > Total cost   (ROI > 0%)
  2. ΔCSAT ≥ −0.1                 (don't make satisfaction worse)
  3. Break-even ≤ 24 months       (recovers cost within 2 years)

Scale to 50% traffic if:
  4. p_value(improvement) < 0.05  (statistically significant gain in A/B test)
  5. No tier exceeds max escalation rate cap
```

**Calculated values at 10,000 users:**

| Metric | Value | Threshold | Pass? |
|--------|-------|-----------|-------|
| Annual savings | ₹22,800 | > total cost | Depends on team size |
| ROI | ~15% | > 0% | ✅ |
| Payback period | ~18 months | ≤ 24 months | ✅ |
| Break-even users | ~8,000 | < current base | ✅ |

---

## 4. What a Production System Would Need

This proof-of-concept makes several simplifications that would need to be addressed in production:

### 4.1 Data

| Need | Current | Production |
|------|---------|-----------|
| Escalation labels | Proxy from text + metadata | CRM system handoff records |
| Customer tier | Topic-based heuristic | Live CRM/billing data |
| Real churn labels | Assumed (B1–B4) | CRM churn events linked to support tickets |
| CSAT scores | Proxy from sentiment | Post-interaction CSAT surveys |

### 4.2 Model

| Component | Current | Production |
|-----------|---------|-----------|
| Feature extraction | Rule-based sentiment + heuristics | Fine-tuned embedding model |
| Reward | Hand-crafted economic formula | Calibrated against real churn data |
| Online learning | Batch retrain | Streaming updates (Online LinUCB) |
| Exploration | Training-time only | Continuous exploration budget |

### 4.3 Infrastructure

| Component | What's needed |
|-----------|--------------|
| Latency | < 100ms for routing decision (LinUCB achieves this easily) |
| Fallback | If model errors → rule-based fallback (ship the rule-based agent too) |
| Monitoring | Track: escalation rate per tier, reward signal drift, CSAT |
| Logging | Every routing decision must be logged for offline evaluation |
| A/B framework | Shadow mode first → 5% traffic → 50% traffic → full |

### 4.4 Ethical Considerations

| Issue | Mitigation |
|-------|-----------|
| **Tier discrimination** | The model may deprioritise Free-tier users. Monitor CSAT across tiers separately. Set a floor: no tier can be fully ignored. |
| **Feedback loops** | If the model always routes complex issues to humans, it never learns from them. Enforce an exploration budget. |
| **Proxy label bias** | Escalation proxies may over-fit to English-language phrasing. Multi-language support needed. |

---

## 5. Metrics Hierarchy for Business Review

When presenting results, use this hierarchy:

**Primary (business outcome):**
- Expected monthly cost reduction (₹)
- Tier-stratified escalation rates vs. caps
- CSAT score change (Δ)

**Secondary (operational):**
- Correct routing rate (true positives + true negatives)
- Missed escalation rate (false negatives — the dangerous error)
- Unnecessary escalation rate (false positives — the wasteful error)

**Tertiary (model health):**
- Reward curve convergence
- Exploration rate over time
- Feature importance stability

> **Rule:** Never report only the tertiary metrics in a business review. Stakeholders care about costs and satisfaction, not loss curves.

---

## 6. Relationship to Course Topics

| AML Topic | Where it appears in this project |
|-----------|----------------------------------|
| **Business framing** | Tier economics, CLV, escalation threshold derivation |
| **Cost-sensitive learning** | Asymmetric reward: missed escalation ≠ unnecessary escalation |
| **Exploration vs exploitation** | UCB and Thompson Sampling exploration parameters |
| **Model evaluation beyond accuracy** | No accuracy anywhere; all metrics are business-aligned |
| **Causal reasoning** | Churn elasticity assumption: correlation (bad experience → churn) vs. causation |
| **Sensitivity analysis** | CSA cost, churn rate, and inference cost sweeps |
| **Deployment constraints** | Latency (< 100ms), capacity (K=50 slots), monitoring |
| **Ethics / fairness** | Tier-stratified CSAT floors; exploration budget |

---

*Back: [Rewards & Escalation](reward_and_escalation.md) | Start: [Overview](overview.md)*
