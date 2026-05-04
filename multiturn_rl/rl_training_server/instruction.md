# Long PPO Training With Local Qwen + Simulation_4

This is the simple server pipeline:

```text
Simulation_4 simulator
  + local fine-tuned Qwen model from Qwen2.5-7B-Instruct-merged/
  + PPO RL agent
  -> learn a policy that increases resolution/reward and reduces escalation
```

There is no Ollama, no vLLM, no localhost endpoint, and no web server.

## 1. Folder Layout

Run everything from:

```bash
cd /data/interns/studentiotlab/AdaptiveBandit/AdaptiveBandit
```

Expected folders:

```text
AdaptiveBandit/
  Qwen2.5-7B-Instruct-merged/
  Simulation_4/
  rl_training_server/
```

Keep outputs organized like this:

- `Simulation_4/artifacts/`: simulator calibration files and PPO artifacts.
- `rl_training_server/runs/`: `nohup` logs, PID files, configs, smoke-test reports, and `latest` symlink.

## 2. Activate Environment

```bash
source /data/interns/studentiotlab/AdaptiveBandit/.venv311/bin/activate
```

Check core packages:

```bash
python - <<'PY'
import torch, transformers, stable_baselines3, gymnasium
print("python ok")
print("cuda:", torch.cuda.is_available())
PY
```

Check the model folder:

```bash
ls Qwen2.5-7B-Instruct-merged/config.json
ls Qwen2.5-7B-Instruct-merged/tokenizer.json
```

## 3. Smoke Test The Local Qwen Model

This loads `Qwen2.5-7B-Instruct-merged` directly with `transformers`.

```bash
python rl_training_server/scripts/smoke_local_qwen.py
```

Output is saved to:

```text
rl_training_server/runs/smoke/local_qwen_behavior_*.jsonl
```

Good signs:

- The model loads without an endpoint.
- Responses are customer-only.
- The output includes:

```text
<response>...</response>
<behavior>{...}</behavior>
```

## 4. Smoke Test Simulator + Local Qwen + NLP Observation

```bash
python rl_training_server/scripts/smoke_simulator_nlp.py
```

This checks:

```text
SupportEnv(nlg_enabled=True)
  -> NLPObservationWrapper
  -> RewardShapedWrapper
  -> ActionMaskedEnv
```

Output is saved to:

```text
rl_training_server/runs/smoke/simulator_nlp_*.json
```

Good signs:

- Observation shape is `[9]`.
- Customer utterances are generated.
- Rewards are not all identical.
- Escalation is masked in early turns.

## 5. Start Long Training With nohup

Recommended launcher:

```bash
bash rl_training_server/scripts/start_rl_training.sh
```

Defaults:

- `TIMESTEPS=5000000`
- `N_ENVS=1`
- `OUTPUT_SUBDIR=rl_qwen_long`
- `SUPPORT_SIM_LOCAL_MODEL_PATH=Qwen2.5-7B-Instruct-merged`
- `HEARTBEAT_FREQ_STEPS=25`
- `METRICS_EVAL_FREQ=500`

For a longer run:

```bash
TIMESTEPS=20000000 OUTPUT_SUBDIR=rl_qwen_20m bash rl_training_server/scripts/start_rl_training.sh
```

Important: keep `N_ENVS=1`. Direct local Qwen loading with multiple workers can load the model multiple times and exhaust GPU memory.

## 6. Manual nohup Command

The launcher above is preferred. This is the equivalent manual form:

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)
mkdir -p rl_training_server/runs/$RUN_ID/logs rl_training_server/runs/$RUN_ID/pids
ln -sfn "$(pwd)/rl_training_server/runs/$RUN_ID" rl_training_server/runs/latest

SUPPORT_SIM_LLM_BACKEND=local \
SUPPORT_SIM_LOCAL_MODEL_PATH=Qwen2.5-7B-Instruct-merged \
SUPPORT_SIM_LLM_MODEL=local-qwen \
SUPPORT_SIM_INTENT_MODEL=local-qwen \
SUPPORT_SIM_AGENT_MODEL=local-qwen \
PYTHONUNBUFFERED=1 \
nohup python rl_training_server/scripts/train_rl_qwen.py \
  --model-path Qwen2.5-7B-Instruct-merged \
  --timesteps 5000000 \
  --output-subdir rl_qwen_long \
  --run-id "$RUN_ID" \
  --n-envs 1 \
  --customer-model local-qwen \
  --intent-model local-qwen \
  --agent-model local-qwen \
  > rl_training_server/runs/$RUN_ID/logs/train.log 2>&1 &

echo $! > rl_training_server/runs/$RUN_ID/pids/rl_training.pid
```

## 7. Monitor Training

```bash
bash rl_training_server/scripts/status_rl_training.sh
```

The status command shows:

- process PID
- GPU usage
- `heartbeat.json` with percent complete, steps/second, elapsed time, and ETA
- latest training log lines
- latest `metrics.csv` rows once available

Live log:

```bash
tail -f rl_training_server/runs/latest/logs/train.log
```

Direct heartbeat:

```bash
cat Simulation_4/artifacts/rl_qwen_long/runs/*/heartbeat.json
```

Stop:

```bash
bash rl_training_server/scripts/stop_rl_training.sh
```

## 8. Output Locations

Run-control files:

```text
rl_training_server/runs/<run_id>/
  config.json
  logs/train.log
  pids/rl_training.pid
```

PPO artifacts:

```text
Simulation_4/artifacts/rl_qwen_long/
  models/
    best_model.zip
    baseline_beating_model.zip
    final_model.zip
  tensorboard/
  training_log.json
  training_summary.json
  runs/<run_id>/
    metrics.csv
    summary.json
    env_config.json
    plots/
      reward_curve.png
      escalation_rate_curve.png
```

## 9. Resume Training

```bash
CHECKPOINT=Simulation_4/artifacts/rl_qwen_long/models/best_model.zip \
OUTPUT_SUBDIR=rl_qwen_long_continue \
TIMESTEPS=5000000 \
bash rl_training_server/scripts/start_rl_training.sh
```

## 10. What Not To Use

Do not use these for the direct local-Qwen pipeline:

- Ollama
- vLLM
- localhost model APIs
- `server_output/runscripts/start_phase10_bg.sh`
- `server_output/runscripts/status_phase10_bg.sh`
- old Phase10 background scripts

The only required runtime folders are:

```text
Qwen2.5-7B-Instruct-merged/
Simulation_4/
rl_training_server/
```
