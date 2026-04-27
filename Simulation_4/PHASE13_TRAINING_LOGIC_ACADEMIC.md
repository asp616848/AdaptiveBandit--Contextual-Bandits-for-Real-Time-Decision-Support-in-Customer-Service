# Phase 13 Training Logic: Academic Technical Note

## 1. Problem Formulation

Phase 13 is a contextual reinforcement learning system for customer-support decision making, trained in a stochastic simulator with language-mediated observations. The agent optimizes a discounted return objective over episodic conversations.

We model the process as an MDP/POMDP hybrid:

- Hidden simulator state: $x_t$
- Policy observation: $o_t$
- Discrete action: $a_t \in \{0,1,2,3,4\}$
- Reward: $r_t$
- Transition kernel: $x_{t+1} \sim P(\cdot \mid x_t, a_t)$

The action mapping is:

- $0$: AskInfo
- $1$: ProvideSolution
- $2$: AffectiveRepair
- $3$: Escalate
- $4$: Close

The training objective is standard discounted return maximization:

$$
J(\pi_\theta)=\mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T-1} \gamma^t r_t\right], \quad \gamma=0.99.
$$

## 2. Simulator Architecture and Coupling

The training stack composes the simulator and wrappers as:

1. `SupportEnv` (core stochastic simulator, optional customer NLG on)
2. `NLPObservationWrapper` (LLM-derived semantic observation + agent response generation)
3. `RewardShapedWrapper` (potential-based shaping)
4. `ActionMaskedEnv` (safety mask on invalid early escalation)
5. `Monitor` (episode statistics)

### 2.1 Core simulator (`SupportEnv`)

`SupportEnv` is the grounded dynamics engine built on:

- `StateEngine`: transition probabilities and action dynamics
- `RewardEngine`: per-turn and terminal economics-aware reward
- `NLGLayer`: customer utterance generation conditioned on transition outcome
- `LumoRAG`: scenario + policy retrieval context

The simulator samples, at episode reset:

- subflow (issue type), from calibrated empirical weights
- customer tier/member level
- persona latent variables $(\rho,\sigma,\tau)$
- value weight, initial information, frustration, and derived progress

### 2.2 RAG/NLG integration

At reset, `LumoRAG` produces scenario/policy context. During each step:

1. policy action is chosen,
2. agent text is generated (via `AgentResponseGenerator`),
3. simulator applies state transition,
4. `NLGLayer` generates the customer utterance based on action, outcome, frustration, and revealed slots,
5. conversation history is updated and fed back into observation construction.

This creates two-sided conversational trajectories instead of static templates.

## 3. Existing Simulator State and Exposed Observation

## 3.1 Full internal state (`SupportEnv.state`)

The simulator tracks at least:

- task context: `subflow`, `subflow_idx`, `tier`, `tier_idx`, `member_level`, `difficulty`
- customer economics: `value_weight`
- persona latents: `rho`, `sigma`, `tau`, `persona_label`
- progress variables: `information`, `progress`, `frustration`, `failed_streak`, `turn_count`
- terminal flags: `resolved`, `escalated`, `dropped_off`, `done`
- optional RAG labels: `lumo_label`, `display_name`

## 3.2 Native simulator observation (base env)

Before NLP wrapping, `SupportEnv` emits a 9D normalized numeric vector:

$$
o_t^{base} = [\text{subflow\_id},\text{tier\_id},d_t,i_t,p_t,f_t,\text{streak}_t,\text{turn}_t,\text{resolved}_t].
$$

## 3.3 Phase 13 policy observation (NLP wrapper)

In Phase 13, policy input is replaced by NLP-derived 9D features:

1. intent_norm
2. confidence
3. sentiment_norm
4. suggested_action_norm
5. escalation_flag
6. info_completeness
7. turn_count_norm
8. history_depth_norm
9. customer_len_norm

So the policy does not directly consume `tier`, `difficulty`, or `failed_streak` as explicit channels.

## 4. Transition Dynamics (Methods and Equations)

Let $i_t$ be information, $p_t$ progress, $f_t$ frustration, and $s_t$ failed streak.

### 4.1 AskInfo

Information-gain occurrence probability:

$$
P(\text{gain})=\mathrm{clip}\left(p_{info}(0.6+0.8\rho),0,1\right).
$$

If gain occurs:

$$
\Delta i_t = \mathrm{clip}\left(\mu_{persona}(1-0.4d)+\epsilon, 0, 1-i_t\right), \quad \epsilon\sim\mathcal{N}(0,\sigma_{ask}^2).
$$

Updates:

$$
i_{t+1}=\mathrm{clip}(i_t+\Delta i_t,0,1), \quad
p_{t+1}=\mathrm{clip}(p_t+0.05\Delta i_t,0,1).
$$

Frustration update:

$$
\Delta f_t = 0.03(1-\rho)+0.01d-0.06|\Delta i_t|+0.03\max(s_t-1,0),
$$
$$
f_{t+1}=\mathrm{clip}(f_t+\Delta f_t,0,1).
$$

### 4.2 ProvideSolution

Success probability (logistic model):

$$
P_{succ}=\sigma\left(\theta_0+\theta_i i_t+\alpha_{subflow}+\alpha_{action}\right).
$$

If successful:

$$
p_{t+1}=\mathrm{clip}\left(p_t+\mathrm{clip}(0.25+0.20i_t,0,1-p_t),0,1\right),
$$
$$
f_{t+1}=\mathrm{clip}\left(f_t-\mathrm{clip}(0.20+0.10i_t,0,f_t),0,1\right),\; s_{t+1}=0.
$$

If $p_{t+1}$ exceeds an auto-resolve threshold, the episode resolves.

If failed:

$$
s_{t+1}=s_t+1,
$$
$$
\Delta f_t=0.05+0.15\sigma+0.10\left(\frac{s_{t+1}}{s_{t+1}+3}\right)(1-\tau),
$$
$$
f_{t+1}=\mathrm{clip}(f_t+\Delta f_t,0,1).
$$

### 4.3 AffectiveRepair

Repair effectiveness probability:

$$
P_{repair}=\mathrm{clip}(p_{repair,base}+0.4\rho,0,1).
$$

If effective: frustration decreases by $0.15(0.6+0.6\sigma)$, else increases by $0.02$.

### 4.4 Escalate

Immediate terminal transition with escalation flag set.

### 4.5 Close

Close score:

$$
\text{score}_{close}=0.45P_{succ}+0.35p_t+0.20i_t-0.25f_t.
$$

Episode resolves iff score exceeds a threshold; otherwise terminal unresolved-close.

### 4.6 Autonomous dropout and timeout

At each non-terminal step, dropout probability is:

$$
P_{drop}=\sigma\left(c_0+c_f f_t+c_s s_t+c_t\,\text{turn}_t+c_\tau\tau\right).
$$

After transition, turn count increments; timeout occurs at $T_{max}=20$ if still non-terminal.

## 5. Reward Design

Total reward is per-turn penalty plus optional terminal component:

$$
r_t = r_{turn} + \mathbb{1}_{terminal}\, r_{terminal}.
$$

Per-turn penalty:

$$
r_{turn}=-\lambda_{turn},\quad \lambda_{turn}=0.15.
$$

Economics-aware terminal core:

$$
r_{econ}=-\omega\, p_{churn}^{terminal}\, V(\text{tier},w), \quad \omega\approx 0.1667.
$$

Customer value-at-risk:

$$
V(\text{tier},w)=V_{base}(\text{tier})\left(1+\kappa w\right), \quad \kappa=0.5.
$$

Terminal churn assignment:

- success: $p_{churn}^{terminal}=0$
- dropout: $p_{churn}^{terminal}=1$
- escalation/timeout/other: logistic churn from $(f,s,turn,\tau)$

Outcome adjustments:

- success bonus: $+\eta_{success}$, with $\eta_{success}=5.0$
- escalation cost: subtract tier-specific escalation cost (enterprise may get bonus)
- unresolved close: additional penalty

Final reward is clipped to $[-5,5]$.

## 6. Potential-Based Reward Shaping

Phase 13 uses policy-invariant potential shaping (enabled by default):

$$
r'_t = r_t + \gamma\Phi(x_{t+1}) - \Phi(x_t).
$$

Potential function:

$$
\Phi(x)=w_i i + w_p p - w_f f,
$$

with weights derived from shaping coefficients in `RewardShaper`.

Because shaping is pure potential form under the same discount $\gamma$, optimal policies are preserved while credit assignment is accelerated.

## 7. PPO Algorithm and Update Equations

The optimizer is PPO (Stable-Baselines3) with policy/value networks (MLP, Tanh) and configuration:

- network: [64, 32]
- `n_steps=2048`, `batch_size=64`, `n_epochs=10`
- `gamma=0.99`, `gae_lambda=0.95`
- `clip_range=0.2`, `ent_coef=0.02`, `vf_coef=0.5`
- `max_grad_norm=0.5`, `learning_rate=1e-4`

### 7.1 Advantage estimation (GAE)

Temporal-difference residual:

$$
\delta_t = r_t + \gamma V_\phi(o_{t+1}) - V_\phi(o_t).
$$

Generalized advantage:

$$
\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}, \quad \lambda=0.95.
$$

Return target:

$$
\hat{R}_t = \hat{A}_t + V_\phi(o_t).
$$

### 7.2 Clipped surrogate policy objective

Probability ratio:

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid o_t)}{\pi_{\theta_{old}}(a_t\mid o_t)}.
$$

PPO clipped objective:

$$
L^{CLIP}(\theta)=\mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\,\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\right)\right],\quad \epsilon=0.2.
$$

Full optimization combines policy, value, and entropy terms:

$$
L(\theta,\phi)=L^{CLIP} - c_v L_V + c_e H[\pi_\theta],
$$

with $c_v=0.5$ and $c_e=0.02$ in this stage.

## 8. Action Masking and Safety Constraint

`ActionMaskedEnv` disables action 3 (Escalate) for the first 3 turns:

$$
\text{mask}_t(3)=0 \text{ if turn}<3,\;1\text{ otherwise}.
$$

If the policy emits an invalid action under the mask, it is replaced with action 0 (AskInfo). This prevents immediate-escalation degenerate behavior in early training.

## 9. Curriculum and Evaluation Method

Curriculum (`CurriculumScheduler`) gradually expands allowable subflows by estimated difficulty:

- Stage easy: from 0 timesteps, bounded difficulty
- Stage medium: from 100k timesteps
- Stage full: from 300k timesteps (all subflows)

Callbacks implement:

- rolling metrics (reward, episode length, resolution/escalation/dropout/timeout)
- periodic deterministic evaluation (mean reward and terminal rates)
- best-model checkpointing
- baseline-beating checkpointing

## 10. How the Policy Uses the Simulator in Practice

At each step:

1. policy receives NLP observation derived from conversation text,
2. policy selects one of 5 support actions,
3. action-conditioned agent text is generated,
4. simulator state transitions with calibrated stochastic dynamics,
5. customer utterance is generated by NLG,
6. reward is computed (base + terminal + shaping),
7. new NLP observation is re-derived from updated dialogue.

Thus, the simulator remains the causal environment, while the policy interface is language-semantic and partially observable.

## 11. Notes on Observability and Leakage

Phase 13 intentionally avoids direct exposure of key hidden simulator internals (e.g., tier, failed streak) in policy input. The classifier invocation also passes an empty subflow string in NLP observation construction. This design pushes the policy toward realistic inference from dialogue context rather than privileged state access.

## 12. Outputs and Artifacts

A typical run writes:

- model checkpoints (`best_model`, `baseline_beating_model`, `final_model`)
- tensorboard logs
- `training_log.json` (metrics/evaluation histories)
- `training_summary.json` (run-level summary)

These artifacts support both optimization monitoring and scientific reproducibility.
