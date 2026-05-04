# RL Training Server

This folder is the run-control layer for long PPO training on the server.

No Ollama. No vLLM. No localhost API server. No model port.

The training process loads the fine-tuned Qwen folder directly with `transformers` and runs:

```text
Simulation_4 simulator + local Qwen customer model + PPO agent
```

Run from the server repo root:

```bash
cd /data/interns/studentiotlab/AdaptiveBandit/AdaptiveBandit
```

Expected layout:

```text
AdaptiveBandit/
  Qwen2.5-7B-Instruct-merged/
  Simulation_4/
  rl_training_server/
```

Storage policy:

- Simulator calibration inputs and PPO artifacts stay under `Simulation_4/artifacts/`.
- `rl_training_server/runs/` stores only `nohup` logs, PID files, smoke-test reports, configs, and the `latest` symlink.
- Do not duplicate `Simulation_4/artifacts` inside this folder.

Main commands:

```bash
python rl_training_server/scripts/smoke_local_qwen.py
python rl_training_server/scripts/smoke_simulator_nlp.py
bash rl_training_server/scripts/start_rl_training.sh
bash rl_training_server/scripts/status_rl_training.sh
bash rl_training_server/scripts/stop_rl_training.sh
```

Long training is launched with `nohup` by `start_rl_training.sh`, so it keeps running after SSH disconnects.
Status includes a heartbeat with percent done, steps/second, and ETA.

PPO outputs are written to:

```text
Simulation_4/artifacts/rl_qwen_long/
```
