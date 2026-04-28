# Quick-Reference Commands

Copy-paste commands to run or test any part of the pipeline individually.
All commands are run from the **repo root**.

---

## Full Pipeline (all three approaches)

```bash
# Full run — NLP skipped automatically if ollama is not running
bash run.sh

# Full run, skip NLP explicitly
SKIP_NLP=1 bash run.sh

# Full run with custom step count
NUM_TIMESTEPS=1000000 SKIP_NLP=1 bash run.sh
```

---

## [1] Numerical Multi-Turn RL  (PPO, state-only, no LLM needed)

```bash
# Default: 1 million steps
bash scripts/train_numerical.sh

# Quick smoke-test (≈2 min)
NUM_TIMESTEPS=10000 bash scripts/train_numerical.sh

# Custom steps + eval episodes
NUM_TIMESTEPS=500000 EVAL_EPISODES=100 bash scripts/train_numerical.sh

# More parallel envs (speeds up training if CPU has cores to spare)
NUM_TIMESTEPS=1000000 N_ENVS=4 bash scripts/train_numerical.sh
```

Output lands in:
- `output/numerical-multi-turn/`  — plots, logs, model zips
- `best_model/numerical/`         — best_model.zip + final_model.zip

---

## [2] NLP Multi-Turn RL  (PPO + LLM-generated customer utterances)

### With ollama (default)

```bash
# Requires: ollama running + model pulled
# ollama serve  (in a separate terminal)
# ollama pull llama3

bash scripts/train_nlp.sh

# Use a different ollama model
OLLAMA_MODEL=mistral bash scripts/train_nlp.sh

# Custom ollama endpoint (remote server)
OLLAMA_ENDPOINT=http://192.168.1.10:11434/v1 OLLAMA_MODEL=llama3 bash scripts/train_nlp.sh
```

### With a local HuggingFace model

```bash
# Point to the merged model directory
HF_MODEL_PATH="Multi-Turn RL/Qwen2.5-7B-Instruct-merged" bash scripts/train_nlp.sh

# Absolute path variant
HF_MODEL_PATH="/home/user/models/Qwen2.5-7B" bash scripts/train_nlp.sh
```

### Quick smoke-test (NLP)

```bash
NUM_TIMESTEPS=10000 OLLAMA_MODEL=llama3 bash scripts/train_nlp.sh
```

Output lands in:
- `output/nlp-multi-turn/`  — plots, logs, model zips
- `best_model/nlp/`         — best_model.zip + final_model.zip

---

## [3] Contextual Bandit  (stub placeholder)

```bash
bash scripts/train_bandit.sh
```

Output lands in:
- `output/contextual-bandit/`

---

## Export / Plot Only  (re-run after training is done)

Re-generate plots and reports from existing artifact files without retraining:

```bash
# Numerical
python tools/export_multiturn_results.py \
  --artifacts-root "Multi-Turn RL/Simulation_4/artifacts" \
  --run-subdir run_numerical \
  --out-dir output/numerical-multi-turn

# NLP
python tools/export_multiturn_results.py \
  --artifacts-root "Multi-Turn RL/Simulation_4/artifacts" \
  --run-subdir run_nlp \
  --out-dir output/nlp-multi-turn
```

---

## Windows Local Testing

On Windows, `python3` may not exist. Override the Python executable:

```bash
# Using a conda environment
PYTHON_BIN="C:/ProgramData/miniconda3/envs/bandit/python.exe" bash scripts/train_numerical.sh

# Or activate the conda env first, then run (bash via Git Bash or WSL)
PYTHON_BIN=python bash scripts/train_numerical.sh
```

---

## Environment Variables — Full Reference

| Variable         | Default                          | Description                              |
|------------------|----------------------------------|------------------------------------------|
| `NUM_TIMESTEPS`  | `1000000`                        | PPO training steps                       |
| `EVAL_EPISODES`  | `200`                            | Episodes used for final evaluation       |
| `N_ENVS`         | `1`                              | Parallel training environments (SubprocVecEnv) |
| `SKIP_NLP`       | `0`                              | Set `1` to skip NLP run entirely         |
| `OLLAMA_MODEL`   | `llama3`                         | Ollama model tag to use for NLG          |
| `OLLAMA_ENDPOINT`| `http://localhost:11434/v1`      | Ollama API base URL                      |
| `HF_MODEL_PATH`  | _(empty, uses ollama)_           | Local HuggingFace model directory        |
| `PYTHON_BIN`     | `python3`                        | Python executable path                   |
