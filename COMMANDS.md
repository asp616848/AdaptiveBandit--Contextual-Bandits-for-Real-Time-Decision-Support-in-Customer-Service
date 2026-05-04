# Quick-Reference Commands

Copy-paste commands to run or test any part of the pipeline individually.
All commands are run from the **repo root**.

---

## Full Pipeline (all three approaches)

```bash
# Full run — NLP skipped automatically if no LLM available
bash run.sh

# Full run, skip NLP explicitly
SKIP_NLP=1 bash run.sh

# Full run with custom step count
NUM_TIMESTEPS=1000000 SKIP_NLP=1 bash run.sh
```

---

## [1] Numerical Multi-Turn RL  (PPO, state-only, no LLM)

```bash
# Default: 1 million steps
bash scripts/train_numerical.sh

# Quick smoke-test (~2 min)
NUM_TIMESTEPS=10000 bash scripts/train_numerical.sh

# Custom steps + eval episodes
NUM_TIMESTEPS=1500000 EVAL_EPISODES=200 bash scripts/train_numerical.sh

# More parallel envs (speeds up training if CPU has cores to spare)
NUM_TIMESTEPS=1500000 N_ENVS=4 bash scripts/train_numerical.sh
```

Output:
- `output/numerical-multi-turn/`  — plots, logs, model zips
- `best_model/numerical/`         — best_model.zip + final_model.zip

---

## [2] NLP Multi-Turn RL  (PPO + LLM-generated customer utterances)

### With a HuggingFace model (hub download)

```bash
# Downloads model on first run (~14 GB for Qwen 7B)
HF_MODEL_PATH=abhi6168/ABCD_CustomerAgent_Qwen_2.5_7b bash scripts/train_nlp.sh

# With custom step count
NUM_TIMESTEPS=100000 HF_MODEL_PATH=abhi6168/ABCD_CustomerAgent_Qwen_2.5_7b bash scripts/train_nlp.sh

# Smoke-test (1 step — just verifies model loads and pipeline runs)
NUM_TIMESTEPS=1 HF_MODEL_PATH=abhi6168/ABCD_CustomerAgent_Qwen_2.5_7b bash scripts/train_nlp.sh
```

### With a local model directory

```bash
HF_MODEL_PATH=/path/to/model bash scripts/train_nlp.sh
```

### With ollama

```bash
# Requires: ollama running + model pulled
# ollama serve  (separate terminal)
# ollama pull llama3

bash scripts/train_nlp.sh

OLLAMA_MODEL=mistral bash scripts/train_nlp.sh

# Remote ollama endpoint
OLLAMA_ENDPOINT=http://192.168.1.10:11434/v1 OLLAMA_MODEL=llama3 bash scripts/train_nlp.sh
```

Output:
- `output/nlp-multi-turn/`  — plots, logs, model zips
- `best_model/nlp/`         — best_model.zip + final_model.zip

---

## [3] Contextual Bandit  (stub placeholder)

```bash
bash scripts/train_bandit.sh
```

Output: `output/contextual-bandit/`

---

## Export / Plot Only  (re-run after training without retraining)

```bash
# Numerical
python tools/export_multiturn_results.py \
  --artifacts-root "multiturn_rl/simulation/artifacts" \
  --run-subdir run_numerical \
  --out-dir output/numerical-multi-turn

# NLP
python tools/export_multiturn_results.py \
  --artifacts-root "multiturn_rl/simulation/artifacts" \
  --run-subdir run_nlp \
  --out-dir output/nlp-multi-turn
```

---

## Windows Local Testing

On Windows, `python3` may not exist. Override the Python executable:

```bash
# Using a conda environment
PYTHON_BIN="C:/ProgramData/miniconda3/envs/bandit/python.exe" bash scripts/train_numerical.sh

# Or activate the conda env first (bash via Git Bash or WSL)
PYTHON_BIN=python bash scripts/train_numerical.sh
```

---

## Environment Variables — Full Reference

| Variable | Default | Description |
|---|---|---|
| `NUM_TIMESTEPS` | `1000000` | PPO training steps |
| `EVAL_EPISODES` | `200` | Episodes used for final evaluation |
| `N_ENVS` | `1` | Parallel training environments |
| `SKIP_NLP` | `0` | Set `1` to skip NLP run entirely (used in run.sh) |
| `HF_MODEL_PATH` | _(empty)_ | HF hub repo ID or local path; if set, uses HF backend |
| `OLLAMA_MODEL` | `llama3` | Ollama model tag (only used when HF_MODEL_PATH is empty) |
| `OLLAMA_ENDPOINT` | `http://localhost:11434/v1` | Ollama API base URL |
| `PYTHON_BIN` | `python3` | Python executable path |
