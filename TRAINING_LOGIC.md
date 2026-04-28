# Training Logic — AdaptiveBandit Multi-Turn RL

Reference for the reward function, algorithm, curriculum, and output artifacts.

---

## Environment

**File**: `multiturn_rl/simulation/env/support_env.py`

| Property | Value |
|---|---|
| Observation space | Box(9,) in [0.0, 1.0] — public features only |
| Action space | Discrete(5) |
| Max turns per episode | 20 (`T_max`) |

### Actions

| ID | Name | Description |
|---|---|---|
| 0 | AskInfo | Request more information from customer |
| 1 | ProvideSolution | Offer a solution or resolution |
| 2 | AffectiveRepair | Acknowledge frustration, de-escalate emotionally |
| 3 | Escalate | Hand off to human agent (terminal) |
| 4 | Close | End the conversation (terminal) |

### Observation Features (public mode)

| Index | Feature | Range |
|---|---|---|
| 0 | sentiment | [0, 1] — inferred from dialogue tone |
| 1 | frustration_trend | [0, 1] — rising vs falling frustration |
| 2 | frustration | [0, 1] — current customer frustration level |
| 3 | info_proxy | [0, 1] — estimated information gathered |
| 4 | failed_streak / 5 | [0, 1] — consecutive failed attempts |
| 5 | turn_count / T_max | [0, 1] — episode progress |
| 6 | last_action_norm | [0, 1] — previous action taken |
| 7 | last_success | [0, 1] — whether last action succeeded |
| 8 | resolved | {0, 1} — issue resolved flag |

The agent sees only features inferrable from natural language — no hidden simulator state (tier, subflow type, difficulty).

---

## Reward Function

**File**: `multiturn_rl/simulation/env/reward_engine.py`

### Per-Turn Reward

```
r_turn = -lambda_turn = -0.15
```

Discourages unnecessarily long conversations.

### Terminal Reward

```
r_terminal = -omega * p_churn * V(tier) + outcome_bonus
```

Where:
- `omega = 1/6` — churn loss weight
- `p_churn` — sigmoid probability customer churns after this outcome
- `V(tier) = base_value[tier] * (1 + kappa * value_weight)` — customer lifetime value

| Tier | base_value | typical V |
|---|---|---|
| Free (guest) | 1.6 | ~1.6 |
| Pro (bronze) | 3.4 | ~4.5 |
| Business (silver) | 7.5 | ~10.5 |
| Enterprise (gold) | 12.0 | ~18.0 |

### Churn Model

```
logit(p_churn) = -3.8 + 3.0*frustration + 0.1*failed_streak + 0.08*turn_count - 1.5*tau
```

`tau` is the customer's failure tolerance (persona parameter). High tau = more tolerant.

### Outcome Bonuses

| Outcome | Bonus |
|---|---|
| `success` | `+eta = +5.0` and `p_churn = 0` |
| `dropout` | `p_churn = 1.0` (worst churn) |
| `timeout` | `p_churn` from churn model |
| `unresolved_close` | `-1.0` additional penalty |
| `escalation` | contextual cost (see below) |

All rewards clipped to `[-5.0, 5.0]`.

### Escalation Reward (Key Fix)

Escalation is **contextually rewarded** — the cost scales down when escalation is the right call:

```
appropriateness = frustration * 2.0 + min(failed_streak * 0.5, 1.5)
effective_cost  = max(base_cost[tier] - appropriateness, 0.0)
r_escalation    = -omega * p_churn * V - effective_cost + enterprise_bonus
```

| Tier | base_cost | When escalation becomes net-positive |
|---|---|---|
| Free | 4.0 | frustration ≥ 0.85 + streak ≥ 3 |
| Pro | 2.0 | frustration ≥ 0.7 + streak ≥ 2 |
| Business | 0.5 | frustration ≥ 0.5 + streak ≥ 1 |
| Enterprise | 0.0 | always + 1.0 bonus |

**Design intent**: unnecessary escalation (low frustration, no failures) is still penalized; escalation when the bot is genuinely stuck (high frustration, repeated failures) is rewarded. Resolution (+5.0) is always preferred when achievable.

---

## Algorithm

**File**: `multiturn_rl/simulation/training/train_ppo.py`

| Hyperparameter | Value |
|---|---|
| Algorithm | PPO (Proximal Policy Optimization) |
| Policy network | MlpPolicy — [128, 128, 64], Tanh activation |
| n_steps | 2048 |
| batch_size | 256 |
| n_epochs | 10 |
| gamma (discount) | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| learning_rate | 3e-4 |

### Reward Shaping

Potential-based shaping applied as a wrapper (`RewardShapedWrapper`):

```
shaping = gamma * Phi(s') - Phi(s)
Phi(s)  = 0.05 * info_gain + 0.10 * progress - frustration_penalty
```

This is Lagrangian-compatible — does not change the optimal policy.

### Action Masking

`ActionMaskedEnv` blocks Escalate (action 3) in the first 3 turns to prevent a degenerate "escalate immediately" policy — **unless** the customer is already in crisis (`frustration > 0.75` AND `failed_streak ≥ 2`), in which case immediate escalation is allowed.

---

## Curriculum Learning

**File**: `multiturn_rl/simulation/training/curriculum.py`

| Stage | Timesteps | Subflow difficulty |
|---|---|---|
| easy | 0 – 100k | ≤ 0.85 (simplest subflows) |
| medium | 100k – 300k | ≤ 1.20 (adds intermediate) |
| full | 300k+ | no filter (all subflows) |

Difficulty is computed from mean action count across the ABCD dataset, normalized to [0.7, 1.6].

---

## Output Artifacts

After a successful run, `output/<approach>/` contains:

### JSON Logs

| File | Contents |
|---|---|
| `training_log.json` | Per-eval metrics: timesteps, mean_reward_100ep, resolution_rate, escalation_rate |
| `training_summary.json` | Best eval reward, final resolution rate, escalation rate, runtime |
| `full_pipeline_report.json` | Combined training + evaluation + baseline comparison |
| `demo_rollouts.json` | 5 example episode traces with action sequence and rewards |

### Plots

| File | Description |
|---|---|
| `plots/reward_curve.png` | Mean reward (100-ep rolling) vs training steps |
| `plots/resolution_curve.png` | Resolution rate + escalation rate vs training steps |
| `plots/eval_curve.png` | Eval reward and resolution rate (dual axis) |
| `plots/mean_reward_leaderboard.png` | PPO vs baselines (random, document_guided) |
| `plots/terminal_outcomes.png` | Episode outcome distribution (success/escalation/dropout/timeout) |

### Models

| File | Description |
|---|---|
| `models/best_model.zip` | Checkpoint with highest eval reward |
| `models/final_model.zip` | Policy after all training steps |

---

## Baselines

Evaluated in `full_pipeline_report.json` under `evaluation.baselines`:

| Policy | Strategy |
|---|---|
| `random` | Uniform random over valid actions |
| `document_guided` | Rule-based: follows subflow script from RAG documents |
| `always_escalate` | Escalates on turn 4 every episode |
| `always_resolve` | Always tries ProvideSolution |

---

## NLG Mode

When `--nlg-enabled` is passed, an LLM generates realistic customer utterances at each turn via `NLGLayer`. The PPO agent's numeric observation is unchanged — NLG only affects the env's internal state transitions (frustration dynamics become language-conditioned).

**Backends**:
- `ollama`: OpenAI-compatible endpoint (default model: `llama3`)
- `hf`: HuggingFace model loaded via `transformers` (default: `abhi6168/ABCD_CustomerAgent_Qwen_2.5_7b`)
