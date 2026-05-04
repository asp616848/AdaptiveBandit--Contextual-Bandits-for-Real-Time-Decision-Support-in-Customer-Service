# Phase13 Server Runbook (Latest Code)

## 1) Current live run (just restarted)

- Server process: `phase13_train.py`
- Current PID file: `/data/interns/studentiotlab/AdaptiveBandit/AdaptiveBandit/server_output/pids/phase13.pid`
- Current PID: `36513`
- Current log: `/data/interns/studentiotlab/AdaptiveBandit/AdaptiveBandit/server_output/logs/phase13_train_20260416_185812.log`
- Launch mode used: `--intent-model llama3 --n-envs 1`

Reason: tinyllama is not available in Ollama model list on this server right now, and DNS to external registries is unstable.

---

## 2) Paths of the new/updated files

Repository root on server:

`/data/interns/studentiotlab/AdaptiveBandit/AdaptiveBandit`

New/updated training pipeline files used by latest stage runs:

- `Simulation_4/scripts/phase11_train.py`
- `Simulation_4/scripts/phase12_train.py`
- `Simulation_4/scripts/phase13_train.py`
- `Simulation_4/env/intent_classifier.py`
- `Simulation_4/env/agent_response_generator.py`
- `Simulation_4/training/nlp_observation.py`
- `Simulation_4/training/text_observation.py`
- `Simulation_4/training/action_masking.py`
- `Simulation_4/training/train_ppo.py`
- `Simulation_4/training/reward_shaping.py`
- `Simulation_4/training/callbacks.py`
- `Simulation_4/training/curriculum.py`
- `Simulation_4/env/support_env.py`
- `Simulation_4/env/state_engine.py`
- `Simulation_4/env/reward_engine.py`
- `Simulation_4/rag/lumo_rag.py`
- `Simulation_4/rag/retriever.py`
- `Simulation_4/artifacts/phase 6/reward_model.json`

Notes:
- `Simulation_4/rag/retriever.py` now includes offline-safe lexical fallback if sentence-transformer cannot be downloaded.
- This avoids hard crash during HF DNS outages.

---

## 3) Complete training logic (Phase13)

Entry script:

- `Simulation_4/scripts/phase13_train.py`

High-level flow:

1. Build `SupportEnv` with `nlg_enabled=True`.
2. Wrap with `NLPObservationWrapper`:
   - Converts conversation into 9D NLP observation.
   - Uses `IntentClassifier` to derive semantic features.
   - Uses `AgentResponseGenerator` to produce agent text before each env step.
3. Wrap with `RewardShapedWrapper` (potential-based shaping).
4. Wrap with `ActionMaskedEnv` (blocks early escalate policy collapse).
5. Wrap with `Monitor` for episode statistics.
6. Train PPO with MLP policy (`[64, 32]`, Tanh).
7. Periodically evaluate and save best model.
8. Save final model and JSON logs/summary.

PPO config used in Phase13:

- `n_steps=2048`
- `batch_size=64`
- `n_epochs=10`
- `gamma=0.99`
- `gae_lambda=0.95`
- `clip_range=0.2`
- `ent_coef=0.02`
- `learning_rate=1e-4`

Callbacks:

- `TrainingMetricsCallback`: rolling episode metrics and resolution/escalation/dropout/timeout rates.
- `BestModelCallback`: periodic evaluation and best checkpoint save.
- `CurriculumCallback`: unlocks harder subflows by timestep.

---

## 4) How it uses the existing simulator

Core simulator is still the same `SupportEnv` + `StateEngine` + `RewardEngine`.

- `SupportEnv` controls episode state, transitions, terminal outcomes, and info dict.
- `StateEngine` computes transition dynamics:
  - AskInfo info gain
  - ProvideSolution stochastic success/failure
  - AffectiveRepair frustration modulation
  - Escalate terminal path
  - Close decision and terminal mapping
  - autonomous dropout and timeout
- `RewardEngine` computes per-turn and terminal economics-aware reward.
- `NLGLayer` generates customer utterances, so conversation is dynamically generated.
- `LumoRAG` injects company docs context into prompts and scenario grounding.

So Stage13 is not a new simulator; it is a new observation/action-language interface on top of the existing calibrated simulator.

---

## 5) Reward design (what is rewarded)

Per-step reward:

- Constant turn penalty: `-lambda_turn` (default `-0.15`) to encourage shorter resolutions.

Terminal reward logic:

- Base economic term:
  - `-omega * p_churn_terminal * V(tier, value_weight)`
- Success bonus:
  - `+eta_success` (default `+5.0`)
- Escalation:
  - subtract tier-specific escalation cost (Free=6.0, Pro=3.0, Business=0.5, Enterprise=0.0)
  - Enterprise can get escalation bonus
- Unresolved close:
  - additional penalty

Reward shaping wrapper:

- Uses potential-based shaping (`gamma*Phi(s') - Phi(s)`) with info/progress/frustration terms.
- This is policy-invariant shaping (strict potential mode), i.e., speeds learning without changing optimal policy class.

Final reward clipping:

- Clipped to `[-5, +5]` in env step.

---

## 6) No-leakage guarantee (important)

Policy observation in Stage13 does NOT expose hidden simulator internals directly.

`NLPObservationWrapper` outputs 9D:

1. intent_norm
2. confidence
3. sentiment_norm
4. suggested_action_norm
5. escalation_flag
6. info_completeness
7. turn_count_norm (derived from conversation length)
8. history_depth_norm (conversation-derived)
9. customer_len_norm (conversation-derived proxy)

Key no-leakage safeguards:

- No direct readout of hidden state fields like `tier_norm` or `failed_streak_norm` in Stage13 observation.
- Classifier call intentionally passes `subflow=""` so hidden flow label is not leaked into classification prompt.
- Features are generated from conversation text + policy context, which mirrors realistic observability.

---

## 7) Components, inputs, outputs

Main components:

- Env core: `SupportEnv`, `StateEngine`, `RewardEngine`
- RAG: `LumoRAG`, `Retriever`, `DocumentStore`, `ScenarioGenerator`
- NLP wrappers: `IntentClassifier`, `AgentResponseGenerator`, `NLPObservationWrapper`
- RL trainer: PPO + callbacks + curriculum + action masking + reward shaping

Training inputs:

- Artifacts under `Simulation_4/artifacts`:
  - transition calibration, success model, persona profiles, reward/tier config
- RAG docs under `Simulation_4/rag/documents`
- Ollama model(s): currently `llama3` available
- CLI flags (important):
  - `--timesteps`
  - `--intent-model`
  - `--agent-model`
  - `--n-envs`
  - `--no-curriculum`
  - `--no-shaping`

Training outputs:

- PPO models (best/final)
- TensorBoard event logs
- `training_log.json`
- `training_summary.json`

---

## 8) Concerned folders/files quick map

- Runtime scripts/logs/pids:
  - `server_output/runscripts/`
  - `server_output/logs/`
  - `server_output/pids/`
- Training code:
  - `Simulation_4/scripts/`
  - `Simulation_4/training/`
- Simulator core:
  - `Simulation_4/env/`
- RAG:
  - `Simulation_4/rag/`
- Artifacts/results:
  - `Simulation_4/artifacts/`

---

## 9) Stop, start, and status commands (copy/paste)

From server repo root:

`cd /data/interns/studentiotlab/AdaptiveBandit/AdaptiveBandit`

Stop current Phase13:

`pkill -f "Simulation_4/scripts/phase13_train.py" || true`

Start latest code (same as current launch):

`mkdir -p server_output/logs server_output/pids; TS=$(date +%Y%m%d_%H%M%S); LOG=server_output/logs/phase13_train_${TS}.log; nohup /data/interns/studentiotlab/AdaptiveBandit/AdaptiveBandit/.venv311/bin/python Simulation_4/scripts/phase13_train.py --intent-model llama3 --n-envs 1 > "$LOG" 2>&1 & echo $! > server_output/pids/phase13.pid; echo PID=$(cat server_output/pids/phase13.pid); echo LOG=$LOG`

Check status (process):

`ps -fp $(cat server_output/pids/phase13.pid)`

Run Phase13 insight dashboard (recommended):

`bash /data/interns/studentiotlab/AdaptiveBandit/AdaptiveBandit/server_output/runscripts/status_phase13_insight.sh`

Watch the insight dashboard every 15s:

`watch -n 15 'bash /data/interns/studentiotlab/AdaptiveBandit/AdaptiveBandit/server_output/runscripts/status_phase13_insight.sh'`

Check latest logs:

`LOG=$(ls -1t server_output/logs/phase13_train_*.log | head -n1); tail -n 120 "$LOG"`

Watch live logs:

`LOG=$(ls -1t server_output/logs/phase13_train_*.log | head -n1); tail -f "$LOG"`

Check for training metric lines quickly:

`LOG=$(ls -1t server_output/logs/phase13_train_*.log | head -n1); grep -nE "time/|rollout/|train/|total_timesteps|fps|ep_rew_mean|approx_kl" "$LOG" | tail -n 30`

Important: Phase13 is LLM-heavy, so logs can be sparse between callback checkpoints (`eval_freq=5000`).
The insight script is designed to show heartbeat, PID/GPU status, and progress estimates even when metric lines are not printing yet.

---

## 10) Where logs and models are saved

Logs while running:

- `/data/interns/studentiotlab/AdaptiveBandit/AdaptiveBandit/server_output/logs/phase13_train_*.log`

PID file:

- `/data/interns/studentiotlab/AdaptiveBandit/AdaptiveBandit/server_output/pids/phase13.pid`

TensorBoard logs:

- `/data/interns/studentiotlab/AdaptiveBandit/AdaptiveBandit/Simulation_4/artifacts/phase13/tensorboard/`

Model checkpoints:

- Best model: `/data/interns/studentiotlab/AdaptiveBandit/AdaptiveBandit/Simulation_4/artifacts/phase13/models/best_model.zip`
- Baseline-beating model (if achieved): `/data/interns/studentiotlab/AdaptiveBandit/AdaptiveBandit/Simulation_4/artifacts/phase13/models/baseline_beating_model.zip`
- Final model (end of training): `/data/interns/studentiotlab/AdaptiveBandit/AdaptiveBandit/Simulation_4/artifacts/phase13/models/final_model.zip`

Training reports:

- `/data/interns/studentiotlab/AdaptiveBandit/AdaptiveBandit/Simulation_4/artifacts/phase13/training_log.json`
- `/data/interns/studentiotlab/AdaptiveBandit/AdaptiveBandit/Simulation_4/artifacts/phase13/training_summary.json`

---

## 11) Important operational caveats

- The server currently shows intermittent DNS failures to HuggingFace/Ollama registries.
- `Retriever` has been patched with lexical fallback, so RAG init does not crash if embedding model download fails.
- `tinyllama` cannot be pulled until DNS/registry access is restored or a valid offline Ollama manifest+blob bundle is imported.
- Current stable run uses llama3 for intent + customer/agent text generation.
