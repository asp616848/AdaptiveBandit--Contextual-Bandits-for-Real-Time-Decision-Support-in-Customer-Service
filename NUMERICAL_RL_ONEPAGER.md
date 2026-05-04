# Numerical Multi-turn RL (state-only PPO) — one-pager

## What this model is
A multi-turn customer-support simulator is treated as an episodic RL environment. The **numerical** model trains a PPO agent using **state-only observations** (no LLM / NLG), learning when to ask for info, propose solutions, repair sentiment, escalate, or close.

- Entry runner: [run.sh](run.sh)
- Numerical training script: [scripts/train_numerical.sh](scripts/train_numerical.sh)
- Full pipeline (train + eval + demos): [multiturn_rl/simulation/scripts/phase10_full_pipeline.py](multiturn_rl/simulation/scripts/phase10_full_pipeline.py)

## Simulator (MDP)
**Environment:** `SupportEnv` (Gymnasium)

- Env definition + episode loop: [multiturn_rl/simulation/env/support_env.py](multiturn_rl/simulation/env/support_env.py)
- Transition dynamics: [multiturn_rl/simulation/env/state_engine.py](multiturn_rl/simulation/env/state_engine.py)
- Reward model: [multiturn_rl/simulation/env/reward_engine.py](multiturn_rl/simulation/env/reward_engine.py)

**Episode / horizon**
- Max turns: `T_max = 20` (timeout terminal)
- Terminals: `success`, `escalation`, `dropout`, `timeout` (plus `unresolved_close` mapped to a penalty)

**Action space (Discrete(5))**
- `0 AskInfo`, `1 ProvideSolution`, `2 AffectiveRepair`, `3 Escalate`, `4 Close`  (see `ACTION_NAMES`)

**Latent state variables (core ones)**
- Subflow + difficulty: `subflow`, `difficulty`
- Customer persona parameters: `rho`, `sigma`, `tau`
- Interaction state: `information`, `progress`, `frustration`, `failed_streak`, `turn_count`
- Termination flags: `resolved`, `escalated`, `dropped_off`, `done`

**Observation vector (shape = (9,))**
Training uses `observation_mode="full"` (see training section). The env also supports `observation_mode="public"` (proxy features a real agent could infer), but that is primarily intended for text/NLG settings.

- “full” features (normalized): subflow id, tier id, difficulty, information, progress, frustration, failed_streak, turn_frac, resolved
- “public” features: sentiment, frustration trend, frustration, info_proxy, failed_streak, turn_frac, last_action, last_success, resolved

**Stochastic dynamics (high-level)**
- `AskInfo`: sometimes increases `information` (persona-conditional), slightly changes `progress`, updates `frustration`
- `ProvideSolution`: succeeds with $p_\text{success}=\sigma(\theta_0 + \theta_i\,\text{information} + \alpha_\text{subflow} + \alpha_\text{action})$; on success increases `progress` and decreases `frustration`, else increases `failed_streak` and `frustration`
- `AffectiveRepair`: probabilistically reduces `frustration`
- `Escalate`: ends episode with `terminal_type="escalation"`
- `Close`: ends episode; outcome depends on a “close score” over success-probability/progress/info/frustration

## Reward model
**Where:** [multiturn_rl/simulation/env/reward_engine.py](multiturn_rl/simulation/env/reward_engine.py)

The reward is **step-cost + terminal utility/loss**:

- Per-turn shaping cost: $r_t = -\lambda_\text{turn}$ with `lambda_turn = 0.15`
- Terminal reward combines:
  - success bonus `eta_success` (default `+5`)
  - escalation: adds a small “stuck” bonus and subtracts a tier-dependent escalation cost (capped), encouraging escalation only when warranted
  - unresolved close: fixed penalty `-1`
  - churn loss: terminal loss is proportional to $p_\text{churn}$ and tier value-at-risk

Churn model (terminal-only) is logistic:

$$
 p_\text{churn} = \sigma(c_0 + c_f\,\text{frustration} + c_s\,\text{failed\_streak} + c_t\,\text{turn\_count} + c_{\tau}\,\tau)
$$

and the churn loss is roughly:

$$
 r_\text{churn} = -\omega\,p_\text{churn}\,V(\text{tier},\text{value\_weight})
$$

Reward is clipped to `[-5, 5]` in the environment step.

## Training algorithm (numerical)
**Algo:** PPO (Stable-Baselines3)

- PPO + config: [multiturn_rl/simulation/training/train_ppo.py](multiturn_rl/simulation/training/train_ppo.py)
- PPO implementation: Stable-Baselines3 (`stable_baselines3.PPO`)

Key PPO settings (from `PPO_CONFIG`):
- Policy: `MlpPolicy`, net: `[128, 128, 64]` with `Tanh`
- `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.05`
- `learning_rate=3e-4`, `n_steps=2048`, `batch_size=256`, `n_epochs=10`

**Curriculum learning (subflow difficulty ramp)**
- Scheduler: [multiturn_rl/simulation/training/curriculum.py](multiturn_rl/simulation/training/curriculum.py)
- Callback wiring: [multiturn_rl/simulation/training/callbacks.py](multiturn_rl/simulation/training/callbacks.py)

Stages are based on subflow “difficulty” (derived from historical mean action counts): easy → medium → full.

**Reward shaping (policy-invariant potential shaping)**
- Shaper + wrapper: [multiturn_rl/simulation/training/reward_shaping.py](multiturn_rl/simulation/training/reward_shaping.py)

Training uses potential-based shaping (default `strict_potential=True`):

$$
 r'_t = r_t + \gamma\,\Phi(s_{t+1}) - \Phi(s_t)
$$

with $\Phi(s)$ combining `information`, `progress`, and `frustration`.

**Action masking**
- Wrapper: [multiturn_rl/simulation/training/action_masking.py](multiturn_rl/simulation/training/action_masking.py)

Masking is effectively disabled (all actions always valid) to avoid corrupting PPO buffers.

## Data / scenario inputs
- Dataset used for scenario context (if present): [dataset/abcd_v1.1.json](dataset/abcd_v1.1.json)
- RAG + scenario context plumbing: [multiturn_rl/simulation/rag/lumo_rag.py](multiturn_rl/simulation/rag/lumo_rag.py)

## Results (this workspace run)
Exported outputs are in: [output/numerical-multi-turn/full_pipeline_report.json](output/numerical-multi-turn/full_pipeline_report.json)

Training summary (1,000,000 steps):
- Summary: [output/numerical-multi-turn/training_summary.json](output/numerical-multi-turn/training_summary.json)
- `best_eval_reward = 1.5656`, `baseline_beaten = true` (step `40,000`)
- `final_resolution_rate ≈ 0.544` (training-time metric callback)

Evaluation (200 episodes, deterministic policy):
- PPO: mean reward `-1.775`, resolution `3.5%`, escalation `64.5%`, dropout `32.0%`, mean turns `7.35`
- Best baseline (`document_guided`): mean reward `0.470`, resolution `50.0%`, escalation `15.5%`
- Leaderboard snapshot: [output/numerical-multi-turn/leaderboard.json](output/numerical-multi-turn/leaderboard.json)

Important caveat for interpretation:
- Training uses `observation_mode="full"` (see [multiturn_rl/simulation/training/train_ppo.py](multiturn_rl/simulation/training/train_ppo.py)), but the evaluation helper in [multiturn_rl/simulation/scripts/phase10_full_pipeline.py](multiturn_rl/simulation/scripts/phase10_full_pipeline.py) constructs `SupportEnv(...)` with default `observation_mode="public"`. This train/eval observation mismatch can strongly degrade reported PPO eval metrics.

## Reproduce (numerical)
- End-to-end runner: `bash run.sh` (skips if outputs exist)
- Numerical only: `NUM_TIMESTEPS=1000000 EVAL_EPISODES=200 N_ENVS=1 bash scripts/train_numerical.sh`

## Files touched (quick index)
- Runner/scripts: [run.sh](run.sh), [scripts/common.sh](scripts/common.sh), [scripts/train_numerical.sh](scripts/train_numerical.sh)
- Pipeline/eval: [multiturn_rl/simulation/scripts/phase10_full_pipeline.py](multiturn_rl/simulation/scripts/phase10_full_pipeline.py)
- Env + reward: [multiturn_rl/simulation/env/support_env.py](multiturn_rl/simulation/env/support_env.py), [multiturn_rl/simulation/env/state_engine.py](multiturn_rl/simulation/env/state_engine.py), [multiturn_rl/simulation/env/reward_engine.py](multiturn_rl/simulation/env/reward_engine.py)
- PPO training: [multiturn_rl/simulation/training/train_ppo.py](multiturn_rl/simulation/training/train_ppo.py), [multiturn_rl/simulation/training/reward_shaping.py](multiturn_rl/simulation/training/reward_shaping.py), [multiturn_rl/simulation/training/curriculum.py](multiturn_rl/simulation/training/curriculum.py), [multiturn_rl/simulation/training/callbacks.py](multiturn_rl/simulation/training/callbacks.py)
- Baselines: [multiturn_rl/simulation/validation/baseline_policies.py](multiturn_rl/simulation/validation/baseline_policies.py)
- Exporter: [tools/export_multiturn_results.py](tools/export_multiturn_results.py)
- Outputs: [output/numerical-multi-turn/](output/numerical-multi-turn/)
