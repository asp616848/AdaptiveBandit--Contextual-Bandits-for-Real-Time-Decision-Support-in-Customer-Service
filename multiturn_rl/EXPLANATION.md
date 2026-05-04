# MultiTurn RL System: Complete Technical Documentation

## Overview

Self-improving customer support agent using:
- **Qwen 2.5-7B** (fine-tuned) as realistic customer in simulator  
- **Gym environment** with state/reward tracking
- **PPO (Proximal Policy Optimization)** for policy learning
- **NLP-based observation** for semantic state representation
- **Single GPU** training (N_ENVS=1 to avoid Qwen memory overhead)

Training goal: Learn when to ask, solve, empathize, escalate, or close conversations while maximizing customer satisfaction + minimizing cost + preventing churn.

---

## 1. System Architecture

### 1.1 Components

#### Qwen Customer Model
- **Model:** `abhi6168/ABCD_CustomerAgent_Qwen_2.5_7b` (7B params, fine-tuned on customer service)
- **Role:** Generates realistic customer responses in conversations
- **Loading:** Direct Python (transformers lib), no external Ollama/vLLM server
- **Output:** Customer behaviors (ask_info, accept_solution, escalate_request, complaint, satisfied)
- **Input:** Conversation history + current state (frustration, resolved_items, etc.)

#### Support Environment (Gym-compatible)
Located in `Simulation_4/env/` and `simulation/env/`

**Core Modules:**
- **state_engine.py:** Tracks internal state variables (frustration, information, success probability)
  - Information gain from AskInfo actions
  - Frustration dynamics (decreases with empathy, increases with failures)
  - Success probability (based on information + subflow + problem difficulty)
  - Customer dropout probability (abandonment risk)

- **reward_engine.py:** Computes immediate & terminal rewards
  - Per-turn penalty: -0.15 (encourage efficiency)
  - Terminal rewards: Success (+5.0), Churn penalty, Escalation cost/bonus
  - All rewards clipped to [-5.0, +5.0]

- **nlg_layer.py:** Natural Language Generation
  - Converts agent actions to customer-facing text
  - Fallback templates if Ollama unavailable

- **rag/lumo_rag.py:** Knowledge base retrieval
  - Retrieves relevant documents for agent responses
  - Provides policy context

- **intent_classifier.py:** LLM-based intent extraction
  - Classifies customer intent from text
  - Extracts sentiment, info completeness, urgency
  - Generates 6 NLP features for observation

#### RL Agent (PPO)
- **Algorithm:** Proximal Policy Optimization (PPO)
- **Network:** 3-layer MLP with [128, 128, 64] neurons
  - Input: 9D NLP observation vector
  - Output: 5D softmax (one logit per action)
  - Activation: Tanh
- **Training:** 150,000 timesteps, batch size 32, 5 epochs/update
- **Hardware:** Single GPU (N_ENVS=1 to avoid Qwen duplication)
- **Typical time:** 12-24 hours per training run

---

## 2. Complete MDP Formulation

$$\mathcal{M} = \langle S, A, P, R, \gamma = 0.99, T_{max} = 20 \rangle$$

### 2.1 State Space

**Observation Space (What Agent Sees):** 9D feature vector from LLM intent classifier

$$s_t = [s_0, s_1, \ldots, s_8]^T \in [0,1]^9$$

| Index | Feature | Range | Source | Meaning |
|-------|---------|-------|--------|---------|
| 0 | intent_norm | [0,1] | IntentClassifier | Customer intent type (0-5 mapped to [0,1]) |
| 1 | confidence | [0,1] | IntentClassifier | Classifier confidence in intent |
| 2 | sentiment_norm | [0,1] | IntentClassifier | 0=frustrated, 0.5=neutral, 1=satisfied |
| 3 | suggested_action | [0,1] | IntentClassifier | LLM-recommended action (0=AskInfo, 1=Close) |
| 4 | escalation_flag | {0,1} | IntentClassifier | Escalation needed? (binary) |
| 5 | info_completeness | [0,1] | IntentClassifier | Information provided estimate |
| 6 | turn_count_norm | [0,1] | Wrapper | current_turn / 20 |
| 7 | history_depth_norm | [0,1] | Wrapper | min(len(history), 40) / 40 |
| 8 | customer_len_norm | [0,1] | Wrapper | min(len(last_msg), 300) / 300 |

**Internal State (Not Observed by Agent):** Tracked by StateEngine for transition dynamics

$$\mathbf{x}_t = \{i_t, d, \rho, f_t, s_{fail}, \tau, t, p_{drop}, \text{subflow}, \text{persona}, V_w\}$$

- $i_t \in [0,1]$: Information completeness
- $d \in [0,1]$: Problem difficulty  
- $\rho \in [0,1]$: Customer responsiveness to AskInfo
- $f_t \in [0,1]$: Frustration level
- $s_{fail} \in \{0,1,2,\ldots\}$: Failed attempts streak
- $\tau \in [0,1]$: Time pressure / resolution urgency
- $t \in \{0,1,\ldots,19\}$: Current turn number
- $p_{drop} \in [0,1]$: Dropout probability (computed)
- $\text{subflow}$: Issue category (billing, technical, etc.)
- $\text{persona}$: Customer segment (Standard, VIP, Enterprise)
- $V_w \in [0,1]$: Customer lifetime value weight

### 2.2 Action Space

$$\mathcal{A} = \{0, 1, 2, 3, 4\}$$

| Action | Code | Effect | When Valid |
|--------|------|--------|-----------|
| AskInfo | 0 | Request info, gain knowledge (∆i), increase frustration | Always (if not max depth) |
| ProvideSolution | 1 | Attempt to resolve with $p_{success}$ | When i ≥ threshold |
| AffectiveRepair | 2 | Show empathy (↓frustration) | When f > 0.3 |
| Escalate | 3 | Transfer to human specialist | Once per episode |
| Close | 4 | End conversation | When satisfied |

**Action Masking:** Invalid actions masked out (PPO respects this).

### 2.3 Transition Dynamics

**Information Gain (AskInfo):**
$$p_{gain} = p_{info\_gain} \cdot (0.6 + 0.8\rho)$$
If gain occurs (Bernoulli sample):
$$\Delta i \sim \mathcal{N}(\mu_{ask}, 0.02^2) \cdot (1 - 0.4d)$$
$$i_{t+1} = \text{clip}(i_t + \Delta i, 0, 1)$$

**Frustration Dynamics (all actions):**
$$\Delta f = 0.03(1-\rho) + 0.01d - 0.06|\Delta i| + 0.03\max(s_{fail}-1, 0)$$
$$f_{t+1} = \text{clip}(f_t + \Delta f, 0, 1)$$

**Success Probability (ProvideSolution):**
$$p_{success} = \sigma(\theta_0 + \theta_i i_t + \alpha_{subflow} + \alpha_{action})$$
Where $\sigma(z) = \frac{1}{1+e^{-z}}$ (sigmoid), typical values:
- $\theta_0 \approx -2.5$ (base difficulty)
- $\theta_i \approx 4.0$ (information importance)
- $\alpha_{subflow}, \alpha_{action}$ from calibration artifacts

**Dropout (Customer Abandonment):**
$$p_{dropout} = \sigma(c_0 + c_f f + c_s s_{fail} + c_t t + c_\tau \tau)$$

Default coefficients:
- $c_0 = -5.0$ (baseline: low dropout)
- $c_f = 3.5$ (frustration multiplier)
- $c_s = 0.3$ (failed streak multiplier)
- $c_t = 0.1$ (turn multiplier)
- $c_\tau = -2.0$ (time pressure reduces dropout)

### 2.4 Reward Function (DETAILED)

**CRUCIAL:** All rewards are actual values from code; clipped to [-5.0, +5.0] at episode end.

#### Per-Turn Reward (Every Step)

$$R_{step} = -\lambda_{turn} = -0.15$$

**Motivation:** Encourage fast resolution; lingering costs the company.

#### Terminal Reward (Episode End)

Computed only when episode terminates (success, dropout, escalation, max turns, or close).

$$R_{terminal} = R_{outcome} + R_{churn} + R_{escalation}$$

##### Component 1: Outcome Reward

**Success outcome:**
$$R_{outcome} = \eta = 5.0$$

**Unresolved close:**
$$R_{outcome} = -1.0$$

**Dropout/Timeout (no explicit bonus, absorbed into churn):**
$$R_{outcome} = 0.0$$

##### Component 2: Churn Risk Penalty

Customer lifetime value at risk:
$$V(\text{tier}, w) = V_{base}(\text{tier}) \cdot (1 + \kappa w)$$

Where:
- $V_{base}(\text{Standard}) = 1.0$
- $V_{base}(\text{Enterprise}) = 5.0$ (or configured)
- $\kappa = 0.5$ (value multiplier)
- $w \in [0,1]$ (customer LTV weight)

Churn probability (at terminal state):
$$p_{churn} = \sigma(c_0 + c_f f + c_s s_{fail} + c_t t + c_\tau \tau)$$

Default: $c_0 = -3.8$, $c_f = 3.0$, $c_s = 0.1$, $c_t = 0.08$, $c_\tau = -1.5$

**Churn penalty:**
$$R_{churn} = -\omega \cdot p_{churn} \cdot V(\text{tier}, w)$$

Where $\omega = \frac{1}{6} \approx 0.1667$

**Example:** Enterprise customer ($V=5$), moderate frustration ($f=0.6$), no failed streaks, turn 15:
$$p_{churn} = \sigma(-3.8 + 3.0 \cdot 0.6 + 0 + 0.08 \cdot 15 - 1.5 \cdot 0.5) = \sigma(-0.95) \approx 0.28$$
$$R_{churn} = -\frac{1}{6} \cdot 0.28 \cdot 5 \approx -0.23$$

##### Component 3: Escalation Reward/Penalty

When outcome = "escalation":

**Stuck Score** (how stuck was agent):
$$\text{stuck} = \min(\max(0.5f + 0.15s_{fail} - 0.4, 0), 0.2)$$

- At $f=0.5, s_{fail}=2$: stuck = 0.05
- At $f=0.9, s_{fail}=4$: stuck = 0.2 (capped)

**Human Recovery Reward:**
$$R_{escalation,recovery} = \text{stuck} \cdot \eta = \text{stuck} \cdot 5.0$$

Up to +1.0 when maximally stuck.

**Appropriateness Credit:**
$$\text{appropriateness} = 3.0f + \min(0.7s_{fail}, 2.5)$$

Reduces escalation cost if justified. Example: $f=0.8, s_{fail}=3$:
$$\text{appropriateness} = 2.4 + 2.1 = 4.5$$

**Escalation Cost:**
$$\text{base\_cost} = \text{escalation\_cost}(\text{tier})$$ (tier-specific, e.g., 2.0 for Standard)

$$\text{effective\_cost} = \max(\text{base\_cost} - \text{appropriateness}, 0)$$
$$\text{effective\_cost} = \min(\text{effective\_cost}, 1.5)$$ (capped at 1.5 to prevent extreme penalty)

**Net Escalation Reward:**
$$R_{escalation} = \text{stuck} \cdot 5.0 - \text{effective\_cost}$$

**Example:** Standard tier, $f=0.8, s_{fail}=3, \text{base\_cost}=2.0$:
$$\text{stuck} = \min(\max(4.0 - 0.4, 0), 0.2) = 0.2$$
$$\text{appropriateness} = 2.4 + 2.1 = 4.5$$
$$\text{effective\_cost} = \min(\max(2.0 - 4.5, 0), 1.5) = 0$$
$$R_{escalation} = 0.2 \cdot 5.0 - 0 = +1.0$$ ✅ Correct escalation rewarded!

**Counter-example:** $f=0.2, s_{fail}=0, \text{base\_cost}=2.0$:
$$\text{stuck} = 0$$
$$\text{appropriateness} = 0.6$$
$$\text{effective\_cost} = \min(\max(2.0 - 0.6, 0), 1.5) = 1.4$$
$$R_{escalation} = 0 - 1.4 = -1.4$$ ❌ Premature escalation penalized!

#### Summary: Reward Bounds

**Per-turn:** $-0.15$

**Terminal (best case - success on Enterprise tier):**
- Outcome: +5.0
- Churn: ≈ 0 (p_churn ≈ 0)
- Total: ≈ +5.0

**Terminal (worst case - dropout on Enterprise):**
- Outcome: 0
- Churn: $-\frac{1}{6} \cdot 1.0 \cdot 5 = -0.833$
- Total: ≈ -0.83

**After clipping:** All rewards in range **[-5.0, +5.0]**

User Note: Simulation_4 has reward_min=-50, reward_max=50 configured, but simulation (the main one) clips to [-5.0, 5.0]. The actual training uses [-5.0, 5.0] bounds.

---

## 3. PPO Training

### Policy Network
$$\pi_\theta(a|s) = \text{softmax}(\text{MLP}_\theta(s))$$

**Architecture:** Input 9D → [128, 128, 64] Tanh → Output 5D softmax

### PPO Loss
$$L^{CLIP} = \mathbb{E}_t \left[ \min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]$$

Where $r_t = \frac{\pi_\theta}{\pi_{old}}$ and $\epsilon = 0.2$.

### GAE Advantages
$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_t^V, \quad \delta_t = R_t + \gamma V(s') - V(s)$$

With $\gamma = 0.99$, $\lambda = 0.95$.

### Hyperparameters
| Param | Value |
|-------|-------|
| n_steps | 128 |
| n_epochs | 5 |
| batch_size | 32 |
| learning_rate | 3e-4 |
| entropy_coef | 0.01 |
| vf_coef | 0.25 |
| clip_range | 0.2 |
| grad_norm_clip | 0.5 |
| **total_timesteps** | **150,000** |
| **n_envs** | **1** |

---

## 4. Conversation Flow

**Each turn:**

1. Qwen generates customer message
2. Intent classifier → 6 NLP features  
3. Wrapper adds 3 time/length features → **9D state**
4. Agent predicts action via $\pi_\theta$
5. NLG converts action to text
6. State Engine updates internal state ($f$, $i$, etc.)
7. Reward Engine returns $-0.15$
8. Loop or terminate

**Termination:** Success (+5.0) | Churn (-ω·p·V) | Max 20 turns

---

## 5. Why This Works

1. **Realistic Behavior:** Qwen not fixed rules
2. **Semantic State:** NLP features generalize to real conversations
3. **Balanced Reward:** Success + efficiency + churn risk
4. **Curriculum:** Can gradually increase difficulty
5. **Offline Ready:** Features extractable from text alone (no Qwen at inference time)

---

## 6. What Agent Learns

- ✅ When to ask (information gathering)
- ✅ When to solve (don't rush)
- ✅ When to empathize (manage frustration)
- ✅ When to escalate (recognize limits)
- ✅ When to close (proper timing)

**Policy:** Maximize satisfaction + minimize cost + prevent churn

---

## 7. Metrics Tracked

- **Mean Reward:** Overall agent performance
- **Success Rate:** % issues resolved
- **Escalation Rate:** % escalated
- **Avg Turns:** Efficiency
- **Churn Probability:** Retention

All logged to TensorBoard (`Simulation_4/artifacts/phase10/tensorboard/`).

---

## 8. Summary

System trains customer support agent via:

1. **Qwen simulator** for realistic customer behavior
2. **NLP observation wrapper** for semantic state extraction
3. **PPO algorithm** to learn optimal actions
4. **Reward balancing** success, efficiency, and business value
5. **Policy deployment** with text-extractable features (offline-safe)

Result: Self-improving agent that learns when to ask, solve, empathize, escalate, and close — generalizing to real customer conversations.
