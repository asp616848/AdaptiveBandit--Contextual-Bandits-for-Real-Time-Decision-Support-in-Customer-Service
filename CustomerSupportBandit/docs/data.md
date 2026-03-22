# Data Sources & Simulation

> **TL;DR:** The system can run on three data sources — real Twitter customer service threads, OpenAssistant quality-rated conversations, and procedurally generated synthetic conversations. All three are converted into the same standard format before training.

---

## 1. Real Datasets

### 1.1 Twitter Customer Support (`twcs.csv`)

**What it is:** The Twitter Customer Support corpus — ~2.8 million tweets exchanged between customers and company support accounts (Apple, Amazon, Uber, etc.).

**What we extract:**

```
Raw tweet stream
      │
      ▼
Reconstruct threads (tweet → reply chains)
      │
      ▼
Filter to company↔customer exchanges only
      │
      ▼
Label escalation (see criteria below)
      │
      ▼
Standard conversation dict
```

**Escalation labelling logic** (proxy, since Twitter doesn't have explicit labels):

| Signal | What it means |
|--------|--------------|
| Escalation phrases in text | Customer explicitly asks for human / manager |
| Tweet went unanswered | Company didn't reply → unresolved |
| Conversation length > 8 turns | Issue too complex for short exchange |
| Negative sentiment throughout | Customer never became satisfied |

> **Assumption:** If ≥2 of these signals are present, we label the conversation as `escalation_needed = True`.

**Tier assignment:** Twitter data doesn't have tier information. We simulate it using topic heuristics — billing/outage topics → higher tier, simple FAQ → lower tier.

---

### 1.2 OpenAssistant (`oasst1-train.csv` / `.parquet`)

**What it is:** ~161,000 human-annotated messages in Assistant-style conversations, with **explicit quality ratings** (1–5 stars per message).

**Why it's valuable:** The quality ratings give us a **proxy reward signal** — we can train a reward model on these labels instead of having to define reward manually.

**What we extract:**

```
Message tree (OA is stored as trees, not linear)
      │
      ▼
Walk tree: each root → leaf path = one conversation
      │
      ▼
Aggregate quality scores → conversation-level label
      │
      ▼
Low mean quality (<3.0) → escalation_needed = True
```

**Quality → Escalation mapping:**

| Mean Quality Score | Interpretation | Escalation Label |
|--------------------|---------------|-----------------|
| < 2.5 | Poor responses throughout | `True` |
| 2.5 – 3.5 | Mixed quality | `True` (marginal) |
| > 3.5 | Satisfactory | `False` |

---

## 2. Synthetic Data Generation

**When used:** When real data is unavailable, or to supplement it. Controlled by `SYNTHETIC_N` in the notebook.

**How it works:**

Each synthetic conversation is procedurally generated to match realistic customer support distributions:

```python
# Pseudocode of what generate_synthetic_conversations() does:

for each conversation:
    tier = sample from [Free(60%), Pro(25%), Business+(12%), Enterprise(3%)]
    turns = sample Poisson(λ=4) capped at 15
    issue_type = sample from tier's typical issues
    sentiment_trajectory = simulate_arc(issue_type)
    escalation_needed = compute_from_tier_and_sentiment()
```

**Sentiment arc patterns:**

| Arc type | Pattern | Typical scenario |
|----------|---------|-----------------|
| `resolved` | Neutral → Positive | Simple FAQs, password resets |
| `frustrated_resolved` | Negative → Neutral | Billing issue fixed |
| `escalation` | Negative → More negative | Outage, refund denied |
| `ambiguous` | Neutral throughout | General inquiry |

**Tier-specific issue types:**

| Tier | Typical issues | Escalation base rate |
|------|---------------|---------------------|
| Free | Account setup, basic FAQ | 8% |
| Pro | Billing, integrations | 18% |
| Business+ | Data export, SLA questions | 28% |
| Enterprise | Security, compliance, custom contracts | 38% |

> **Why not just use synthetic?** Synthetic data has correct *statistical* properties but lacks the messy real-world variation in language — run-on sentences, typos, ambiguous phrasing. Real data makes the feature extractor more robust.

---

## 3. Standard Conversation Format

All three sources are converted to this dict before any agent or environment sees them:

```python
{
    "conversation_id":   "twcs_00123",      # Unique ID
    "tier":              "Pro",             # Free | Pro | Business+ | Enterprise
    "num_turns":         5,                 # Number of message pairs
    "turns": [                              # List of turn dicts
        {
            "speaker":  "customer",
            "text":     "My payment failed again",
            "sentiment": -0.4              # [-1, +1]
        },
        {
            "speaker":  "agent",
            "text":     "I can look into that...",
            "sentiment":  0.1
        },
        ...
    ],
    "escalation_needed": True,             # Ground-truth label
    "source":            "twitter",        # twitter | openassistant | synthetic
    "quality_score":     2.8,             # OA quality (if available)
    "issue_type":        "billing",
    "sentiment_final":   -0.3,            # Final turn sentiment
}
```

---

## 4. Feature Engineering (23-Dimensional Vector)

The agent never sees raw text. It sees a **23-dim numeric vector** computed from the conversation state. This is what the bandit uses as its "context".

### Feature Groups

**Group 1: Sentiment (5 features)**

| Feature | Description |
|---------|-------------|
| `sentiment_current` | Sentiment of the latest customer message |
| `sentiment_mean` | Mean sentiment over full conversation |
| `sentiment_trajectory` | Slope of sentiment over time (improving/worsening?) |
| `sentiment_min` | Worst sentiment seen so far |
| `sentiment_variance` | How much sentiment fluctuates |

**Group 2: Complexity (7 features)**

| Feature | Description |
|---------|-------------|
| `num_turns` | Current turn count (normalised to [0, 1]) |
| `avg_message_length` | Average characters per message |
| `escalation_phrase_count` | How many escalation phrases triggered |
| `question_ratio` | Fraction of messages that ask questions |
| `vocabulary_diversity` | Unique word ratio (higher = more complex) |
| `has_numbers` | Does the conversation contain account numbers / order IDs? |
| `response_delay_proxy` | Are there urgency words like "still waiting", "urgent"? |

**Group 3: Escalation Signals (3 features)**

| Feature | Description |
|---------|-------------|
| `escalation_risk_score` | Weighted sum of escalation phrase triggers |
| `unresolved_indicator` | Did previous bot attempts fail? |
| `repeat_contact_indicator` | Has customer contacted multiple times? |

**Group 4: Tier Economics (6 features)**

| Feature | Description |
|---------|-------------|
| `tier_one_hot[0..3]` | One-hot encoding of Free/Pro/Biz+/Enterprise |
| `monthly_profit_norm` | Normalised monthly profit for this tier |
| `clv_norm` | Normalised CLV for this tier |

**Group 5: Turn Position (2 features)**

| Feature | Description |
|---------|-------------|
| `turn_fraction` | Current turn / max turns (progress through conversation) |
| `is_first_turn` | Binary: 1 if turn 0, else 0 |

> **Why 23?** This was chosen to balance expressiveness with the sample-efficiency requirements of linear bandits. LinUCB scales as O(d²) per update — larger d hurts convergence on small datasets.

---

## 5. Train / Eval Split

| Split | Size | Used for |
|-------|------|----------|
| Train | 80% | Agent parameter updates |
| Eval | 20% | Held-out performance measurement |

Conversations are split by ID (not by turn), so there's no leakage of conversation context across splits.

---

*Back: [Overview](overview.md) | Next: [MDP Formulations](mdp_formulation.md)*
