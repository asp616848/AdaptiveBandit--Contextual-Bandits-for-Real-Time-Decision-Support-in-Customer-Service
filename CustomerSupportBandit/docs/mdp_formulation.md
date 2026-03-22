# MDP Formulations — How Each Agent Thinks

> **Key idea:** We model customer support as a decision-making problem. The two phases use different formulations — bandits for simple routing, full MDP for multi-turn planning.

---

## Background: What is a Bandit vs. an MDP?

| Property | Contextual Bandit | MDP (DQN) |
|----------|------------------|-----------|
| Number of decisions | **One** per conversation | **Many** per conversation |
| State transitions | No — action ends episode | Yes — actions change the state |
| Credit assignment | Immediate (1 feedback signal) | Delayed (across turns) |
| Use case | Routing at conversation start | Dialogue management |
| Computational cost | Very low (linear) | Higher (neural net + replay) |

---

## Phase I — Contextual Bandit

### Formal Setup

The bandit problem is defined as:

```
At each conversation t:
  ─ Observe context  x_t ∈ ℝ²³  (the feature vector)
  ─ Choose action    a_t ∈ {bot(0), human(1)}
  ─ Observe reward   r_t ∈ ℝ    (economic reward)
  ─ Update model based on (x_t, a_t, r_t)
```

There is **no next-state** — the conversation ends after the routing decision (or we treat it as a one-step episode).

---

### Agent 1: Rule-Based Baseline

No learning. Hard-coded decision tree based on tier and escalation phrases.

```
if tier == "Free":
    → always bot  (no escalation budget)

if tier == "Pro":
    if escalation_phrases_detected OR sentiment < -0.3:
        → human
    else:
        → bot

if tier == "Business+":
    if sentiment < -0.1 OR turn_count > 3:
        → human  (aggressive escalation)

if tier == "Enterprise":
    → always human  (protect high CLV at all costs)
```

**Strengths:** Completely interpretable, no training needed.
**Weaknesses:** Ignores within-tier variation. A simple Pro billing question is treated the same as a Pro escalation crisis.

---

### Agent 2: LinUCB

**Key idea:** Learn a separate linear reward model for each action, and add an exploration bonus that shrinks as confidence grows (Upper Confidence Bound).

#### How it works

For each action `a`, maintain:
- `A_a` — a d×d matrix (accumulated outer products of context vectors)
- `b_a` — a d-dim vector (accumulated context-weighted rewards)

The learned weight vector is `θ_a = A_a⁻¹ b_a`.

**At decision time:**

```
For each action a:
    predicted_reward  = θ_a · x_t            (exploitation)
    exploration_bonus = α √(x_t · A_a⁻¹ · x_t)  (exploration)
    UCB_score(a)      = predicted_reward + exploration_bonus

Choose a* = argmax UCB_score(a)
```

**After observing reward r_t:**

```
A_a ← A_a + x_t · x_t ᵀ
b_a ← b_a + r_t · x_t
```

**Exploration parameter `α`:** Controls the width of the confidence bound.
- `α = 0` → pure exploitation (greedy)
- `α = 1.0` → balanced (what we use)
- `α → ∞` → always explores

**Why it works well here:**
- Linear models are sample-efficient — 3,000 conversations is enough
- UCB exploration is theoretically optimal: regret grows as O(d√T log T)
- Interpretable: the weight vector `θ` shows which features matter

---

### Agent 3: Thompson Sampling

**Key idea:** Maintain a *probability distribution* over possible reward models, sample from it, and act greedily on the sample. uncertainty shrinks naturally as data accumulates.

#### How it works

Maintain a Bayesian linear regression posterior for each action `a`:

```
Prior:      w_a ~ N(0, I)     (zero mean, identity covariance)

After observations:
  - Posterior mean:  μ_a = (X_a ᵀ X_a + I)⁻¹ X_a ᵀ r_a
  - Posterior cov:   Σ_a = v² (X_a ᵀ X_a + I)⁻¹
    where v² is the noise variance hyperparameter
```

**At decision time:**

```
For each action a:
    Sample  w̃_a ~ N(μ_a, Σ_a)    (draw from posterior)
    Score   s_a  = w̃_a · x_t     (expected reward under this sample)

Choose a* = argmax s_a
```

This naturally balances exploration (high uncertainty regions will occasionally produce very high samples) and exploitation (as uncertainty collapses, samples converge to the mean).

**Parameter `v²` (noise variance):**
- Higher v² → wider posterior → more exploration
- We use v² = 0.5 (moderate exploration)

**Comparison to LinUCB:**

| | LinUCB | Thompson Sampling |
|--|--------|------------------|
| Exploration mechanism | Deterministic UCB bonus | Stochastic posterior sampling |
| Theoretical guarantees | Frequentist regret bounds | Bayesian regret bounds |
| Computational cost | O(d²) per update | O(d²) per update + sampling |
| Practical performance | Tends to over-explore early | Often converges faster |

---

## Phase II — Full MDP (DQN)

### Why we need an MDP here

The bandit treats each conversation as a **single decision**. But for Business+/Enterprise customers, the right action depends on what happened in previous turns:

- Turn 1: Customer reports login issue → ask clarifying question
- Turn 2: Customer confirms MFA is the problem → provide solution
- Turn 3: Customer says it's still broken → escalate

This is inherently **sequential** — the value of escalating at turn 3 depends on what you tried in turns 1 and 2. This is exactly what an MDP captures.

---

### MDP Definition

```
M = ⟨S, A, P, R, γ⟩

S  =  ℝ²³          (same 23-dim feature vector, updated each turn)
A  =  {ask_info, provide_solution, escalate, close}   (4 actions)
P  =  conversation dynamics (implicit — driven by real/synthetic data)
R  =  economic reward function (see reward_and_escalation.md)
γ  =  0.99          (discount factor — almost no discounting for short episodes)
```

### The Four MDP Actions

| Action | When to use | Effect |
|--------|------------|--------|
| `ask_info` | Issue is unclear, more context needed | Extends conversation +1 turn; penalty −0.1 |
| `provide_solution` | Issue is diagnosed, try a fix | If successful: +5 bonus; if failed: −penalty |
| `escalate` | Bot can't handle it | Hands off to human; immediate −escalation_cost |
| `close` | Issue resolved | +5 resolution bonus |

### State Transitions

```
Turn t: state s_t = feature_vector(conversation_up_to_turn_t)

Action a_t selected by DQN
        │
        ▼
Environment simulates customer response:
  - 'ask_info'         → customer provides more detail → s_t+1 updated
  - 'provide_solution' → success with prob p_resolve  → done=True or continue
  - 'escalate'         → conversation ends, human takes over → done=True
  - 'close'            → conversation ends → done=True
```

`p_resolve` is estimated from:
- Current sentiment (`higher sentiment` → `higher p_resolve`)
- Turn count (`more turns` → `lower p_resolve`, issue is complex)
- Tier (`Enterprise issues` → `lower p_resolve` by default)

---

### DQN Architecture

```
State s_t (23-dim)
      │
      ▼
  Linear(23 → 128) + ReLU
      │
  Linear(128 → 64) + ReLU
      │
  Linear(64 → 32)  + ReLU
      │
  Linear(32 → 4)   (one output per action)
      │
      ▼
Q(s_t, ·)  →  select action a_t = argmax Q(s_t, a)
```

Pure NumPy implementation — no PyTorch/TensorFlow dependency.

### Training Mechanics

**Experience Replay:**
Each `(s, a, r, s', done)` tuple is stored in a circular buffer (capacity 50,000). Training randomly samples mini-batches of 64 to break temporal correlations.

**Target Network:**
A second copy of the network (updated every 100 steps) provides stable Q-value targets:

```
Target:  y_t = r_t + γ · max_a Q_target(s_t+1, a)   [if not done]
         y_t = r_t                                     [if done]

Loss:    L = (Q_online(s_t, a_t) - y_t)²
```

**ε-greedy Exploration:**

```
ε starts at 1.0        (random actions — pure exploration)
ε decays by ×0.995     each episode
ε floors at 0.05       (5% random forever to prevent policy lock-in)
```

---

### Decision Flow Comparison

```
Contextual Bandit (Phase I):

  Conversation arrives
         │
  Extract features (23-dim)
         │
  LinUCB/TS selects: BOT or HUMAN
         │
  Single reward observed
         │
  Model updated
         │
  Next conversation


DQN (Phase II):

  Conversation arrives
         │
  Extract features (23-dim)  ← Turn 1 state
         │
  DQN selects action_1 (ask_info / provide_solution / escalate / close)
         │
  Customer responds → new state (turns 2 features)
         │
  DQN selects action_2
         │
  ... (repeat until escalate or close)
         │
  Episode ends — total return = Σ γᵗ rₜ
         │
  Replay buffer updated, network trained
```

---

*Back: [Data](data.md) | Next: [Rewards & Escalation](reward_and_escalation.md)*
