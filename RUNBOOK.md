# Runbook — AdaptiveBandit: Contextual Bandits for Real-Time Decision Support in Customer Service

Complete guide to running, testing, and evaluating all three approaches in this repo.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Requirements](#3-requirements)
4. [Installation](#4-installation)
5. [Running the Full Pipeline](#5-running-the-full-pipeline)
6. [Running Individual Approaches](#6-running-individual-approaches)
   - [Numerical Multi-Turn RL](#61-numerical-multi-turn-rl-non-nlg)
   - [NLP Multi-Turn RL](#62-nlp-multi-turn-rl-nlg-enabled)
   - [Contextual Bandit](#63-contextual-bandit-stub)
7. [Output Structure](#7-output-structure)
8. [Configuration Reference](#8-configuration-reference)
9. [LLM Backend Setup](#9-llm-backend-setup)
10. [Windows Local Testing](#10-windows-local-testing)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Project Overview

This project implements reinforcement learning agents for automated customer support triage:

| Approach | Algorithm | Customer Utterances | Status |
|---|---|---|---|
| **Numerical Multi-Turn RL** | PPO (Proximal Policy Optimization) | State-only (no NLG) | Fully implemented |
| **NLP Multi-Turn RL** | PPO + NLG layer | LLM-generated (ollama or local HF model) | Fully implemented |
| **Contextual Bandit** | LinUCB / Thompson Sampling | — | Stub (planned) |

The RL environment (`SupportEnv`) simulates a 5-action customer support interaction:

| Action ID | Action | Description |
|---|---|---|
| 0 | Greet | Open the conversation |
| 1 | Request Info | Ask the customer for more details |
| 2 | Provide Info | Give a solution or information |
| 3 | Escalate | Transfer to a human agent |
| 4 | Close | End the conversation |

The agent observes only features a real agent could infer from natural language — it cannot see hidden simulator state (subflow type, difficulty, tier, etc.).

---

## 2. Repository Structure

```
.
├── run.sh                          ← Full pipeline (all three approaches)
├── COMMANDS.md                     ← Copy-paste quick-reference
├── RUNBOOK.md                      ← This file
├── requirements.txt                ← Python dependencies
├── scripts/
│   ├── common.sh                   ← Shared setup sourced by all scripts
│   ├── train_numerical.sh          ← Standalone: numerical multi-turn
│   ├── train_nlp.sh                ← Standalone: NLP multi-turn
│   └── train_bandit.sh             ← Standalone: contextual bandit
├── tools/
│   └── export_multiturn_results.py ← Export artifacts → output/ + generate plots
├── Multi-Turn RL/
│   └── Simulation_4/
│       ├── env/
│       │   ├── support_env.py      ← SupportEnv gymnasium environment
│       │   └── nlg_layer.py        ← LLM-backed customer utterance generator
│       ├── training/
│       │   ├── train_ppo.py        ← PPO training loop
│       │   ├── callbacks.py        ← Logging + curriculum callbacks
│       │   └── action_masking.py   ← ActionMaskedEnv wrapper
│       ├── scripts/
│       │   └── phase10_full_pipeline.py  ← Orchestrates train + eval
│       ├── llm/
│       │   └── backends.py         ← ollama / HuggingFace LLM backends
│       └── artifacts/              ← Raw training artifacts (gitignored)
├── output/                         ← Final outputs for submission (gitignored)
│   ├── numerical-multi-turn/
│   ├── nlp-multi-turn/
│   └── contextual-bandit/
└── best_model/                     ← Best model checkpoints
    ├── numerical/
    └── nlp/
```

---

## 3. Requirements

- **OS**: Ubuntu 22.04 (primary target) or Windows 10/11 with Git Bash
- **Python**: 3.9, 3.10, or 3.11
- **Disk**: ~2 GB (dataset + models + venv)
- **RAM**: 8 GB minimum; 16 GB recommended for NLP mode
- **GPU**: Optional; CPU training works fine for numerical mode

For NLP mode (one of):
- [ollama](https://ollama.com) installed and running with a model pulled, **or**
- A local HuggingFace model directory (e.g. `Multi-Turn RL/Qwen2.5-7B-Instruct-merged`)

---

## 4. Installation

The scripts handle installation automatically via Python venv. No manual steps needed.

On Ubuntu 22.04:

```bash
# Ensure python3 and venv are available
sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip

# Clone the repo (if not already)
git clone <repo-url>
cd AdaptiveBandit--Contextual-Bandits-for-Real-Time-Decision-Support-in-Customer-Service

# The first run will create .venv/ and install all dependencies automatically
bash run.sh
```

Dependencies installed automatically from `requirements.txt`:
- `stable-baselines3` — PPO implementation
- `gymnasium` — RL environment interface
- `torch` — neural network backend
- `faiss-cpu` — FAISS vector index for RAG retriever
- `openai` — OpenAI-compatible client used by NLG layer (works with ollama)
- `requests` — HTTP client for ollama availability checks
- `matplotlib`, `numpy`, `pandas` — plotting and numerics

---

## 5. Running the Full Pipeline

```bash
# Run everything (NLP is auto-skipped if ollama is not running)
bash run.sh

# Skip NLP explicitly and run faster
SKIP_NLP=1 bash run.sh

# Smoke-test (short run — ~3 minutes per approach)
NUM_TIMESTEPS=10000 SKIP_NLP=1 bash run.sh

# Production run (1M steps, all approaches)
NUM_TIMESTEPS=1000000 bash run.sh
```

`run.sh` calls the individual scripts in order:
1. `scripts/train_numerical.sh`
2. `scripts/train_nlp.sh` (or skips it)
3. `scripts/train_bandit.sh`

Each sub-script is self-contained and can be re-run independently.

---

## 6. Running Individual Approaches

### 6.1 Numerical Multi-Turn RL (non-NLG)

PPO agent trained with pure numeric observations. No LLM required.

```bash
# Default (1 million steps, ~15–25 min on CPU)
bash scripts/train_numerical.sh

# Quick test (~2 min)
NUM_TIMESTEPS=10000 bash scripts/train_numerical.sh

# With parallel envs (speeds up training)
NUM_TIMESTEPS=1000000 N_ENVS=4 bash scripts/train_numerical.sh
```

**What it does:**
1. Sets up venv and installs dependencies (first run only)
2. Trains PPO for `NUM_TIMESTEPS` steps with curriculum learning (easy → medium subflows)
3. Exports plots, logs, and model files to `output/numerical-multi-turn/`
4. Copies best checkpoint to `best_model/numerical/`

**Expected output files:**

```
output/numerical-multi-turn/
├── training_log.json           ← step-by-step metrics (reward, resolution rate, etc.)
├── training_summary.json       ← best eval reward, final resolution rate, runtime
├── full_pipeline_report.json   ← combined training + eval report
├── demo_rollouts.json          ← example episode traces
├── plots/
│   ├── reward_curve.png        ← mean reward vs training steps
│   ├── resolution_curve.png    ← resolution + escalation rate vs steps
│   ├── eval_curve.png          ← eval reward and resolution rate
│   ├── mean_reward_leaderboard.png  ← PPO vs baselines comparison
│   └── terminal_outcomes.png   ← episode outcome distribution
└── models/
    ├── best_model.zip          ← best checkpoint (by eval reward)
    └── final_model.zip         ← final policy after all training steps
```

---

### 6.2 NLP Multi-Turn RL (NLG-enabled)

PPO agent trained with LLM-generated customer utterances as part of the environment.

#### With ollama (recommended)

```bash
# 1. Start ollama in a separate terminal
ollama serve

# 2. Pull the model (first time only)
ollama pull llama3

# 3. Run training
bash scripts/train_nlp.sh

# Custom model
OLLAMA_MODEL=mistral bash scripts/train_nlp.sh

# Remote ollama server
OLLAMA_ENDPOINT=http://192.168.1.5:11434/v1 OLLAMA_MODEL=llama3 bash scripts/train_nlp.sh
```

#### With a local HuggingFace model

```bash
HF_MODEL_PATH="Multi-Turn RL/Qwen2.5-7B-Instruct-merged" bash scripts/train_nlp.sh
```

The path can be absolute or relative to the repo root. The model directory must contain `config.json` and the model weights (merged format, not PEFT adapter-only).

**Output structure** is the same as numerical, under `output/nlp-multi-turn/` and `best_model/nlp/`.

---

### 6.3 Contextual Bandit (stub)

```bash
bash scripts/train_bandit.sh
```

This creates the expected output directory structure and a status placeholder file. Full implementation is planned (LinUCB / Thompson Sampling over the same 5-action, 9-feature space as the RL environment).

---

## 7. Output Structure

After a full run, the repo root will contain:

```
output/
├── numerical-multi-turn/
│   ├── training_log.json
│   ├── training_summary.json
│   ├── full_pipeline_report.json
│   ├── demo_rollouts.json
│   ├── plots/
│   │   ├── reward_curve.png
│   │   ├── resolution_curve.png
│   │   ├── eval_curve.png
│   │   ├── mean_reward_leaderboard.png
│   │   └── terminal_outcomes.png
│   └── models/
│       ├── best_model.zip
│       └── final_model.zip
├── nlp-multi-turn/               ← same structure (only if NLP ran)
│   └── ...
└── contextual-bandit/
    ├── status.txt
    ├── plots/
    ├── logs/
    └── models/

best_model/
├── numerical/
│   ├── best_model.zip
│   └── final_model.zip
└── nlp/                          ← only if NLP ran
    ├── best_model.zip
    └── final_model.zip
```

---

## 8. Configuration Reference

All options are set via environment variables before calling the script.

| Variable | Default | Description |
|---|---|---|
| `NUM_TIMESTEPS` | `1000000` | Total PPO environment steps |
| `EVAL_EPISODES` | `200` | Episodes used for policy evaluation after training |
| `N_ENVS` | `1` | Parallel training environments (`SubprocVecEnv`) |
| `SKIP_NLP` | `0` | Set `1` to skip the NLP training run |
| `OLLAMA_MODEL` | `llama3` | Ollama model tag for NLG customer utterances |
| `OLLAMA_ENDPOINT` | `http://localhost:11434/v1` | Ollama OpenAI-compatible API base URL |
| `HF_MODEL_PATH` | _(empty)_ | Path to local HuggingFace model; if set, overrides ollama |
| `PYTHON_BIN` | `python3` | Python executable (override on Windows: `PYTHON_BIN=python`) |

**Syntax** (bash):
```bash
NUM_TIMESTEPS=500000 N_ENVS=2 bash scripts/train_numerical.sh
```

---

## 9. LLM Backend Setup

### Option A — ollama (easiest)

```bash
# Install ollama (Ubuntu)
curl -fsSL https://ollama.com/install.sh | sh

# Start the server
ollama serve

# Pull a model (in a new terminal)
ollama pull llama3       # ~4 GB
ollama pull mistral      # ~4 GB
ollama pull phi3         # ~2 GB (smaller / faster)

# Verify
ollama list
```

Then run:
```bash
OLLAMA_MODEL=llama3 bash scripts/train_nlp.sh
```

### Option B — Local HuggingFace model

The repo includes `Multi-Turn RL/Qwen2.5-7B-Instruct-merged` as the default local model path. Verify the directory exists and contains model weights:

```bash
ls "Multi-Turn RL/Qwen2.5-7B-Instruct-merged/"
# Expected: config.json, tokenizer.json, model-*.safetensors (or pytorch_model.bin)
```

Then run:
```bash
HF_MODEL_PATH="Multi-Turn RL/Qwen2.5-7B-Instruct-merged" bash scripts/train_nlp.sh
```

The HF backend loads the model in 8-bit quantization if GPU is available, otherwise runs in float32 on CPU (slow — use ollama on CPU-only machines).

---

## 10. Windows Local Testing

The scripts are written for bash and target Ubuntu 22.04 for submission. For local Windows development:

**Option 1 — Git Bash** (recommended)

```bash
# Open Git Bash in the repo root
# Override PYTHON_BIN if python3 doesn't exist
PYTHON_BIN=python bash scripts/train_numerical.sh
```

**Option 2 — conda environment**

```bash
# In Git Bash, with conda env "bandit" already created
PYTHON_BIN="C:/ProgramData/miniconda3/envs/bandit/python.exe" bash scripts/train_numerical.sh
```

**Option 3 — WSL2 (Ubuntu)**

Run exactly as on Ubuntu — the scripts work without modification.

**Note:** `run.sh` uses `python3` by default, which is the correct executable name on Ubuntu 22.04. On Windows, set `PYTHON_BIN=python` or use the conda env path.

---

## 11. Troubleshooting

### `python3: command not found`
Set `PYTHON_BIN` to your Python 3 executable:
```bash
PYTHON_BIN=python bash scripts/train_numerical.sh
```

### `OMP: Error #15: Initializing libiomp5md.dll`
Already handled — `KMP_DUPLICATE_LIB_OK=TRUE` is set in `scripts/common.sh`. If you still see it, set it manually:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
bash scripts/train_numerical.sh
```

### `ModuleNotFoundError: No module named 'Simulation_4'`
`PYTHONPATH` must include `Multi-Turn RL/`. This is set automatically by `common.sh`. If running scripts directly with Python (not via bash), set it manually:
```bash
export PYTHONPATH="$PWD/Multi-Turn RL"
python "Multi-Turn RL/Simulation_4/scripts/phase10_full_pipeline.py" --help
```

### Ollama connection refused
```bash
# Check if ollama is running
ollama list

# Start it
ollama serve

# Verify the endpoint
curl http://localhost:11434/api/tags
```

### Training crashes mid-run
Artifacts are checkpointed every 10k steps. Restart the same script — the pipeline script's `--continue-from _none_` flag starts fresh. To resume from a checkpoint, pass the checkpoint step:
```bash
# Example: resume from step 400000
python "Multi-Turn RL/Simulation_4/scripts/phase10_full_pipeline.py" \
  --artifacts-root "Multi-Turn RL/Simulation_4/artifacts" \
  --output-subdir run_numerical \
  --timesteps 1000000 \
  --continue-from 400000
```

### Plots not generated / empty
Re-run the export script after training completes:
```bash
python tools/export_multiturn_results.py \
  --artifacts-root "Multi-Turn RL/Simulation_4/artifacts" \
  --run-subdir run_numerical \
  --out-dir output/numerical-multi-turn
```
