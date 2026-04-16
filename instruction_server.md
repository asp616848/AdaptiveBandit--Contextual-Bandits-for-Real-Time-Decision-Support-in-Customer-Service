# Linux GPU Server Run Instructions

This file documents the current remote setup on:
- Host: `10.1.40.63`
- User: `studentiotlab`
- Base path: `/data/interns/studentiotlab/AdaptiveBandit`

## 1) Current Status

- Python 3.11 virtual environment is ready at:
  - `/data/interns/studentiotlab/AdaptiveBandit/.venv311`
- Full pipeline background run completed successfully; you can restart any time with the run script.
- Ollama daemon is installed and running on `127.0.0.1:11434`.
- NLG-enabled simulator check was executed and report saved.

## 2) Important Paths

- Project root:
  - `/data/interns/studentiotlab/AdaptiveBandit/Simulation_4`
- Artifacts root:
  - `/data/interns/studentiotlab/AdaptiveBandit/Simulation_4/artifacts`
- Durable runtime output root:
  - `/data/interns/studentiotlab/AdaptiveBandit/server_output`
- Background scripts:
  - `/data/interns/studentiotlab/AdaptiveBandit/server_output/runscripts`

## 3) Background Training / Evaluation (Persistent)

Scripts created:
- `start_phase10_bg.sh`
- `status_phase10_bg.sh`
- `stop_phase10_bg.sh`

Run:

```bash
cd /data/interns/studentiotlab/AdaptiveBandit
bash server_output/runscripts/start_phase10_bg.sh
```

Check progress:

```bash
bash server_output/runscripts/status_phase10_bg.sh
```

Stop run:

```bash
bash server_output/runscripts/stop_phase10_bg.sh
```

Tail latest live log:

```bash
tail -f /data/interns/studentiotlab/AdaptiveBandit/server_output/logs/latest.log
```

## 4) Where Outputs and Checkpoints Are Written

- Active run subdir:
  - `/data/interns/studentiotlab/AdaptiveBandit/Simulation_4/artifacts/phase10_persist`
- Model checkpoints:
  - `/data/interns/studentiotlab/AdaptiveBandit/Simulation_4/artifacts/phase10_persist/models`
- Symlink to current models:
  - `/data/interns/studentiotlab/AdaptiveBandit/server_output/checkpoints/models_current`
- Symlink to latest run directory:
  - `/data/interns/studentiotlab/AdaptiveBandit/server_output/reports/latest_run_dir`

## 5) Resume After Disconnect / Reboot

Reconnect and run:

```bash
cd /data/interns/studentiotlab/AdaptiveBandit
bash server_output/runscripts/status_phase10_bg.sh
```

If process is not running, restart from the best available checkpoint:

```bash
bash server_output/runscripts/start_phase10_bg.sh
```

The start script auto-selects checkpoint priority:
1. `phase10_persist/models/best_model.zip`
2. `phase10_persist/models/final_model.zip`
3. `phase10_prod_linux2/models/best_model.zip`
4. `phase10_final/models/best_model.zip`
5. `phase10_v3/models/best_model.zip`

## 6) Ollama Management

Scripts created:
- `start_ollama.sh`
- `status_ollama.sh`
- `pull_ollama_model.sh`

Start / verify Ollama:

```bash
cd /data/interns/studentiotlab/AdaptiveBandit
bash server_output/runscripts/start_ollama.sh
bash server_output/runscripts/status_ollama.sh
```

Pull model (when DNS/network is fixed):

```bash
bash server_output/runscripts/pull_ollama_model.sh llama3
```

## 7) NLG-Enabled Simulator Smoke Check

Current check report path:
- `/data/interns/studentiotlab/AdaptiveBandit/server_output/reports/nlg_enabled_check.json`

Current outcome:
- NLG path is active and reaches Ollama endpoint.
- Response currently returns model-not-found because no local model is installed yet.

## 8) Known Blocker and Fix

Current blocker while pulling `llama3`:
- DNS lookup to `registry.ollama.ai` fails via `::1:53` on server.

Observed error pattern:
- `lookup registry.ollama.ai on [::1]:53: connection refused`

After infra/DNS fix, rerun:

```bash
cd /data/interns/studentiotlab/AdaptiveBandit
bash server_output/runscripts/pull_ollama_model.sh llama3
bash server_output/runscripts/status_ollama.sh
```

Then repeat the NLG smoke check by re-running your Python check command or pipeline demo with `nlg_enabled=True`.
