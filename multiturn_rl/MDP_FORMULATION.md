# MDP Formulation: Customer Support RL Agent

## 1. MDP Tuple Definition

The customer support system is formulated as a Markov Decision Process (MDP):

$$\mathcal{M} = \langle S, A, P, R, \gamma, T_{max} \rangle$$

Where:
- **S**: State space (observation space for the agent)
- **A**: Action space (5 discrete support actions)
- **P**: State transition dynamics (simulator + customer behavior)
- **R**: Reward function (immediate + terminal rewards)
- **γ**: Discount factor (0.99)
- **T_max**: Episode maximum length (20 turns)

---

## 2. Observation Space (What the Agent Sees)

### 2.1 NLP Observation Space (Primary)

The agent observes a **9-dimensional feature vector** extracted via LLM-based intent classification:

$$s_t \in \mathbb{R}^9, \quad s_t = [s_0, s_1, \ldots, s_8]^T$$

Where each component is:

| Index | Feature | Range | Meaning |
|-------|---------|-------|---------|
| $s_0$ | `intent_norm` | $[0,1]$ | Classified customer intent (normalized) |
| $s_1$ | `confidence` | $[0,1]$ | LLM confidence in intent classification |
| $s_2$ | `sentiment_norm` | $[0,1]$ | Customer sentiment (0=frustrated, 0.5=neutral, 1=satisfied) |
| $s_3$ | `suggested_action_norm` | $[0,1]$ | LLM-suggested action (0=AskInfo, 1=Close) |
| $s_4$ | `escalation_flag` | $\{0,1\}$ | Binary: does LLM suggest escalation? |
| $s_5$ | `info_completeness` | $[0,1]$ | Estimated info provided by customer |
| $s_6$ | `turn_count_norm` | $[0,1]$ | $\frac{\text{turn\_count}}{T_{max}} = \frac{\text{turn\_count}}{20}$ |
| $s_7$ | `history_depth_norm` | $[0,1]$ | $\frac{\min(\text{history\_length}, 40)}{40}$ |
| $s_8$ | `customer_len_norm` | $[0,1]$ | $\frac{\min(\text{last\_msg\_length}, 300)}{300}$ |

**Observation Space (formal):**
$$\mathcal{S} = [0,1]^9$$

### 2.2 Internal State Space (Simulator)

The environment also maintains an internal state (not directly observed):

$$\mathbf{x}_t = \{i_t, d, \rho, f_t, s_{fail}, \tau, t, p_{drop}, \text{subflow}, \text{persona}\}$$

Where:
- $i_t \in [0,1]$: Information completeness
- $d \in [0,1]$: Problem difficulty
- $\rho \in [0,1]$: Customer responsiveness
- $f_t \in [0,1]$: Frustration level
- $s_{fail} \in \mathbb{Z}_{\geq 0}$: Failed attempts streak
- $\tau \in [0,1]$: Time pressure / resolution urgency
- $t \in \{0,1,\ldots,19\}$: Current turn number
- $p_{drop} \in [0,1]$: Dropout probability
- $\text{subflow} \in \{\text{billing}, \text{technical}, \ldots\}$: Issue category
- $\text{persona} \in \{\text{Standard}, \text{VIP}, \text{Enterprise}, \ldots\}$: Customer segment

---

## 3. Action Space

### 3.1 Discrete Action Set

$$\mathcal{A} = \{0, 1, 2, 3, 4\}$$

| Action | Code | Meaning | Effect |
|--------|------|---------|--------|
| AskInfo | 0 | Request more information | ↑ information, ↑ frustration |
| ProvideSolution | 1 | Offer a solution | Attempt resolution (may fail) |
| AffectiveRepair | 2 | Show empathy | ↓ frustration, ↑ loyalty |
| Escalate | 3 | Transfer to specialist | Transition to human agent |
| Close | 4 | End conversation | Terminal action if customer satisfied |

**Action Space (formal):**
$$\mathcal{A} = \{a \mid a \in \mathbb{Z}, 0 \leq a \leq 4\}$$

### 3.2 Action Masking

Not all actions are valid in every state. The action masking mechanism ensures:

$$\mathcal{A}(s) \subseteq \mathcal{A}$$

For example:
- Cannot close if customer still has unresolved issues
- Cannot escalate twice in same episode
- Cannot offer same solution twice consecutively

---

## 4. Transition Dynamics

### 4.1 State Transition Probability

The environment follows a partially observable, stochastic MDP:

$$P(s_{t+1} | s_t, a_t) = P(\mathbf{x}_{t+1} | \mathbf{x}_t, a_t)$$

Transitions are deterministic given action, but include sources of randomness:

1. **Information gain (AskInfo):**
$$p_{gain} = p_{info\_gain} \cdot (0.6 + 0.8 \rho)$$
If gain occurs:
$$\Delta i = \mathcal{N}(\mu_{ask}, \sigma_{ask}^2) \cdot (1 - 0.4d)$$
$$i_{t+1} = \text{clip}(i_t + \Delta i, 0, 1)$$

2. **Frustration dynamics:**
$$\Delta f = 0.03(1-\rho) + 0.01d - 0.06|\Delta i| + 0.03 \cdot \max(s_{fail}-1, 0)$$
$$f_{t+1} = \text{clip}(f_t + \Delta f, 0, 1)$$

3. **Success probability (ProvideSolution):**
$$p_{success} = \sigma(\theta_0 + \theta_i \cdot i_t + \alpha_{subflow} + \alpha_{action})$$

Where $\sigma(z) = \frac{1}{1+e^{-z}}$ is the sigmoid function, with:
- $\theta_0 \approx -2.5$ (base intercept)
- $\theta_i \approx 4.0$ (information importance)
- $\alpha_{subflow}$: subflow-specific offset
- $\alpha_{action}$: action-specific offset

4. **Dropout (customer abandonment):**
$$p_{dropout} = \sigma(c_0 + c_f f_t + c_s s_{fail} + c_t t + c_\tau \tau)$$

With default coefficients:
- $c_0 = -5.0$
- $c_f = 3.5$ (frustration multiplier)
- $c_s = 0.3$ (streak multiplier)
- $c_t = 0.1$ (turn multiplier)
- $c_\tau = -2.0$ (time pressure multiplier)

---

## 5. Reward Function

### 5.1 Per-Turn Reward (Immediate)

Every step incurs a small cost to encourage efficiency:

$$R_t(\text{step}) = -\lambda_{turn}$$

Where $\lambda_{turn} = 0.15$ (per-turn penalty).

**Interpretation:** Agent must resolve issues quickly; lingering is costly.

### 5.2 Terminal Reward (Episode End)

Upon episode termination, a final reward is calculated based on outcome:

$$R_T(\text{terminal}) = R_{outcome} + R_{churn} + R_{escalation}$$

#### 5.2.1 Outcome Reward

**Success (customer issue resolved):**
$$R_{success} = \eta = 5.0$$

**Unresolved close (agent gave up):**
$$R_{unresolved} = -1.0$$

**Timeout (exceeded T_max turns without resolution):**
$$R_{timeout} = 0 \text{ (absorbed into churn penalty)}$$

#### 5.2.2 Churn Risk Penalty

The system estimates probability customer will churn and penalizes it:

$$p_{churn}(f, s_{fail}, t, \tau) = \sigma(c_0 + c_f f + c_s s_{fail} + c_t t + c_\tau \tau)$$

With coefficients:
- $c_0 = -3.8$
- $c_f = 3.0$
- $c_s = 0.1$
- $c_t = 0.08$
- $c_\tau = -1.5$

**Customer lifetime value (tier-adjusted):**
$$V(\text{tier}, w_{value}) = V_{base}(\text{tier}) \cdot (1 + \kappa \cdot w_{value})$$

Where:
- $V_{base}(\text{Standard}) = 1.0$, $V_{base}(\text{Enterprise}) = 5.0$, etc.
- $\kappa = 0.5$ (value weight multiplier)
- $w_{value} \in [0,1]$ (customer lifetime value weight)

**Churn penalty:**
$$R_{churn} = -\omega \cdot p_{churn} \cdot V(\text{tier}, w_{value})$$

Where $\omega = \frac{1}{6} \approx 0.1667$ (churn risk scaling).

**Interpretation:** Losing a high-value customer is very costly.

#### 5.2.3 Escalation Reward/Penalty

When action = Escalate:

**Stuck score (how stuck was the agent):**
$$\text{stuck\_score} = \min(\max(0.5 f + 0.15 s_{fail} - 0.4, 0), 0.2)$$

**Human recovery reward:**
$$R_{escalation,positive} = \text{stuck\_score} \cdot \eta$$

**Escalation cost (tier-dependent):**
$$\text{base\_cost} = \text{escalation\_cost}(\text{tier})$$

**Appropriateness credit (reduce penalty if justified):**
$$\text{appropriateness} = 3.0 f + \min(0.7 s_{fail}, 2.5)$$

**Effective cost:**
$$\text{effective\_cost} = \max(\text{base\_cost} - \text{appropriateness}, 0)$$
$$\text{effective\_cost} = \min(\text{effective\_cost}, 1.5) \text{ (capped)}$$

**Full escalation reward:**
$$R_{escalation} = \text{stuck\_score} \cdot \eta - \text{effective\_cost}$$

**Interpretation:** Escalating when stuck is good; escalating prematurely is penalized.

#### 5.2.4 Dropout (Customer Leaves)

If customer drops out before episode end:
$$R_{dropout} = -\omega \cdot 1.0 \cdot V(\text{tier}, w_{value}) = -\omega \cdot V$$

(Maximum churn probability = 1.0)

### 5.3 Total Discounted Return

The agent maximizes discounted cumulative reward:

$$G_t = \sum_{k=0}^{T-t} \gamma^k R_{t+k}$$

Where $\gamma = 0.99$ (discount factor).

**Clipping:** All terminal rewards are clipped to $[-5.0, +5.0]$ to ensure stability.

---

## 6. Value Functions & Bellman Equations

### 6.1 State Value Function

$$V^\pi(s) = \mathbb{E}_\pi\left[ G_t \mid S_t = s \right] = \mathbb{E}_\pi\left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k} \mid S_t = s \right]$$

### 6.2 Action Value Function (Q-function)

$$Q^\pi(s,a) = \mathbb{E}_\pi\left[ G_t \mid S_t = s, A_t = a \right]$$

### 6.3 Bellman Expectation Equations

For intermediate steps (non-terminal):
$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]$$

For optimal policy:
$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]$$

---

## 7. PPO Training Algorithm

### 7.1 Policy Parameterization

$$\pi_\theta(a|s) = \text{softmax}(\text{MLP}_\theta(s))$$

**Network architecture:**
- Input: 9-dimensional observation
- Hidden layers: [128, 128, 64] neurons
- Activation: Tanh
- Output: 5-dimensional softmax (one logit per action)

### 7.2 PPO Objective

PPO maximizes:
$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ (probability ratio)
- $\hat{A}_t$ (advantage estimate)
- $\epsilon = 0.2$ (clip range)

### 7.3 Generalized Advantage Estimation (GAE)

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_t^V$$

Where:
- $\delta_t^V = R_t + \gamma V(s_{t+1}) - V(s_t)$ (TD residual)
- $\lambda = 0.95$ (GAE parameter)

### 7.4 Training Configuration

| Hyperparameter | Value | Role |
|---|---|---|
| $n_{steps}$ | 128 | Rollout length per environment |
| $n_{epochs}$ | 5 | Policy update epochs per batch |
| $batch\_size$ | 32 | Minibatch size for gradient updates |
| $\gamma$ | 0.99 | Discount factor |
| $\lambda$ | 0.95 | GAE smoothing parameter |
| $\epsilon$ | 0.2 | Clip range for policy ratio |
| $\eta_{lr}$ | $3 \times 10^{-4}$ | Learning rate |
| $\beta_{ent}$ | 0.01 | Entropy bonus coefficient |
| $\beta_{vf}$ | 0.25 | Value function loss coefficient |
| $\|\nabla\|_{max}$ | 0.5 | Gradient norm clipping |
| $T_{total}$ | 150,000 | Total timesteps to train |
| $n_{envs}$ | 1 | Number of parallel environments (N_ENVS=1 required for Qwen) |

### 7.5 Loss Function

PPO combines three terms:

$$L(\theta) = L^{CLIP}(\theta) - \beta_{ent} H(\pi_\theta) + \beta_{vf} L^{VF}(\theta)$$

Where:
- $L^{CLIP}$: Policy gradient (clipped)
- $H(\pi_\theta)$: Entropy bonus (exploration)
- $L^{VF}$: Value function loss (value prediction MSE)

---

## 8. Episode Termination Conditions

An episode terminates when any of the following occurs:

1. **Success:** Customer problem is resolved
   $$\text{Terminal if: } p_{success} > \text{threshold} \text{ AND customer accepts solution}$$

2. **Dropout/Churn:** Customer abandons conversation
   $$\text{Terminal if: } p_{dropout} > \mathcal{U}(0,1)$$

3. **Escalation:** Agent transfers to human (simulator takeover)
   $$\text{Terminal if: } a_t = 3 \text{ (Escalate action)}$$

4. **Max turns reached:**
   $$\text{Terminal if: } t \geq T_{max} = 20$$

5. **Close action with satisfaction:**
   $$\text{Terminal if: } a_t = 4 \text{ (Close) AND customer satisfied}$$

---

## 9. Qwen Customer Model Integration

### 9.1 Customer Response Generation

At each turn $t$, Qwen generates customer behavior:

$$b_t = \text{Qwen}(\text{prompt}_t, \text{history}_{1:t-1})$$

Where prompt contains:
- Conversation history
- Current state (frustration, resolved issues)
- Persona description
- RAG-retrieved knowledge

**Behavior outputs:** One of {`ask_info`, `accept_solution`, `escalate_request`, `complaint`, `satisfied`}

### 9.2 Intent Classification

After customer message, LLM intent classifier extracts features:

$$\mathbf{f}_{NLP} = \text{IntentClassifier}(\text{history}, \text{context})$$

Outputs 6 features:
- Intent classification (0-5)
- Confidence score
- Sentiment
- Suggested action
- Escalation flag
- Information completeness

These feed directly into observation $s_t$.

---

## 10. Summary: MDP Components

| Component | Value/Range | Notes |
|-----------|------------|-------|
| **Observation Space** | $\mathbb{R}^9 \subset [0,1]^9$ | NLP features from intent classifier |
| **Action Space** | $\{0,1,2,3,4\}$ | 5 discrete support actions |
| **Transition Model** | Stochastic | State engine + Qwen customer |
| **Reward (per-turn)** | $-0.15$ | Constant penalty for efficiency |
| **Reward (success)** | $+5.0$ | Problem resolved |
| **Reward (churn)** | $-\omega \cdot p_{churn} \cdot V$ | Customer abandonment cost |
| **Discount Factor** | $\gamma = 0.99$ | Long-horizon optimality |
| **Episode Length** | $T_{max} = 20$ | Maximum turns per conversation |
| **Horizon** | Finite (episodic) | Reset after terminal state |

---

## 11. Example Trajectory

**Episode Structure:**

$$s_0 \xrightarrow{a_0=-0.15} s_1 \xrightarrow{a_1=-0.15} s_2 \xrightarrow{a_2=-0.15} \ldots \xrightarrow{a_{T-1}=R_{terminal}} \text{DONE}$$

**Concrete example:**
```
Turn 0: s₀ = [0.3, 0.8, 0.2, 0.1, 0, 0.2, 0.0, 0.0, 0.0]  (frustrated customer)
        Agent chooses a=0 (AskInfo) → r=-0.15

Turn 1: s₁ = [0.4, 0.75, 0.15, 0.2, 0, 0.4, 0.05, 0.1, 0.1]  (more info)
        Agent chooses a=2 (AffectiveRepair) → r=-0.15

Turn 2: s₂ = [0.5, 0.8, 0.4, 0.3, 0, 0.5, 0.1, 0.15, 0.15]  (less frustrated)
        Agent chooses a=1 (ProvideSolution) → r=-0.15

Turn 3: Success! Terminal state reached
        Final reward: R_terminal = +5.0 - 0 (no churn) = +5.0

Total return: G₀ = -0.15×(1 + 0.99 + 0.99² + ...) + 5.0×0.99³ ≈ -5.07 + 4.85 ≈ -0.22
```

---

## 12. Key Equations Quick Reference

| Formula | Purpose |
|---------|---------|
| $p_{success} = \sigma(\theta_0 + \theta_i i_t + \alpha_{sf} + \alpha_a)$ | Solution acceptance |
| $p_{dropout} = \sigma(c_0 + c_f f_t + c_s s_{fail} + c_t t + c_\tau \tau)$ | Customer abandonment |
| $p_{churn} = \sigma(-3.8 + 3.0 f + 0.1 s_{fail} + 0.08 t - 1.5 \tau)$ | Terminal churn risk |
| $R_{terminal} = \eta \cdot \mathbb{1}_{success} - \omega p_{churn} V - cost_{escalation}$ | Final reward |
| $V = V_{base}(1 + \kappa w_{value})$ | Customer lifetime value |
| $G_t = \sum_k \gamma^k R_{t+k}$ | Discounted return |

