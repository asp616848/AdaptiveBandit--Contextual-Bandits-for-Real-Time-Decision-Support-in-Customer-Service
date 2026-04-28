# Runbook — AdaptiveBandit: Contextual Bandits for Real-Time Decision Support in Customer Service

Complete guide to running, testing, and evaluating all three approaches.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Requirements](#3-requirements)
4. [Installation](#4-installation)
5. [Running the Full Pipeline](#5-running-the-full-pipeline)
6. [Running Individual Approaches](#6-running-individual-approaches)
   - [Numerical Multi-Turn RL](#61-numerical-multi-turn-rl)
   - [NLP Multi-Turn RL](#62-nlp-multi-turn-rl)
   - [Contextual Bandit](#63-contextual-bandit-stub)
7. [Output Structure](#7-output-structure)
8. [Configuration Reference](#8-configuration-reference)
9. [LLM Backend Setup](#9-llm-backend-setup)
10. [Windows Local Testing](#10-windows-local-testing)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Project Overview

This project implements reinforcement learning agents for automated customer support triage.
See [TRAINING_LOGIC.md](TRAINING_LOGIC.md) for the full reward function, algorithm, and curriculum documentation.

| Approach | Algorithm | Customer Utterances | Status |
|---|---|---|---|
| **Numerical Multi-Turn RL** | PPO | State-only (no NLG) | Fully implemented |
| **NLP Multi-Turn RL** | PPO + NLG | LLM-generated (HF hub or ollama) | Fully implemented |
| **Contextual Bandit** | LinUCB / Thompson Sampling | — | Stub (planned) |

The RL environment (`SupportEnv`) simulates 5-action customer support interactions:

| Action ID | Action | Description |
|---|---|---|
| 0 | AskInfo | Request more details from customer |
| 1 | ProvideSolution | Give a solution or information |
| 2 | AffectiveRepair | Acknowledge frustration, de-escalate |
| 3 | Escalate | Transfer to human agent |
| 4 | Close | End the conversation |

---

## 2. Repository Structure

```
.
├── run.sh                          ← Full pipeline (all three approaches)
├── COMMANDS.md                     ← Copy-paste quick-reference
├── RUNBOOK.md                      ← This file
├── TRAINING_LOGIC.md               ← Reward function, algorithm, curriculum docs
├── requirements.txt                ← Python dependencies (base)
├── requirements-llm.txt            ← LLM dependencies (HF backend only)
├── scripts/
│   ├── common.sh                   ← Shared setup sourced by all scripts
│   ├── train_numerical.sh          ← Standalone: numerical multi-turn
│   ├── train_nlp.sh                ← Standalone: NLP multi-turn
│   └── train_bandit.sh             ← Standalone: contextual bandit
├── tools/
│   └── export_multiturn_results.py ← Export artifacts → output/ + plots
├── multiturn_rl/
│   └── simulation/
│       ├── env/
│       │   ├── support_env.py      ← SupportEnv gymnasium environment
│       │   ├── reward_engine.py    ← Reward model (terminal + per-turn)
│       │   ├── state_engine.py     ← State transition dynamics
│       │   └── nlg_layer.py        ← LLM-backed customer utterance generator
│       ├── training/
│       │   ├── train_ppo.py        ← PPO training loop
│       │   ├── callbacks.py        ← Logging + curriculum callbacks
│       │   ├── action_masking.py   ← ActionMaskedEnv wrapper
│       │   ├── curriculum.py       ← Curriculum scheduler
│       │   └── reward_shaping.py   ← Potential-based shaping wrapper
│       ├── scripts/
│       │   └── phase10_full_pipeline.py  ← Orchestrates train + eval
│       ├── llm/
│       │   └── backends.py         ← HuggingFace / ollama backends
│       ├── rag/                    ← RAG retriever + pre-built FAISS index
│       ├── validation/             ← Pre-train environment validation
│       └── artifacts/              ← Training artifacts (tracked in git)
├── output/                         ← Final outputs for submission
│   ├── numerical-multi-turn/
│   ├── nlp-multi-turn/
│   └── contextual-bandit/
└── best_model/                     ← Best model checkpoints
    ├── numerical/
    └── nlp/
```

---

## 3. Requirements

- **OS**: Ubuntu 22.04 (primary) or Windows 10/11 with Git Bash
- **Python**: 3.9, 3.10, or 3.11
- **Disk**: ~2 GB base; +15 GB if downloading HF model
- **RAM**: 8 GB minimum; 16 GB recommended for NLP mode

For NLP mode (one of):
- HuggingFace model: set `HF_MODEL_PATH` to a HF hub repo ID or local path
- [ollama](https://ollama.com) installed and running with a model pulled

---

## 4. Installation

All scripts handle venv creation and dependency installation automatically. No manual steps needed.

```bash
# Ubuntu 22.04 prerequisites (if not already installed)
sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip

# First run creates .venv/ and installs all dependencies automatically
bash run.sh
```

Base dependencies (`requirements.txt`): `numpy`, `pandas`, `matplotlib`, `gymnasium`, `stable-baselines3`, `faiss-cpu`, `openai`, `requests`

LLM dependencies (`requirements-llm.txt`): `transformers`, `accelerate`, `huggingface_hub`, `sentencepiece` — installed automatically by `train_nlp.sh` when HF backend is selected.

---

## 5. Running the Full Pipeline

```bash
# Run everything (NLP auto-skipped if no LLM available)
bash run.sh

# Skip NLP
SKIP_NLP=1 bash run.sh

# Smoke-test (~3 minutes per approach)
NUM_TIMESTEPS=10000 SKIP_NLP=1 bash run.sh

# With HF NLG model
HF_MODEL_PATH=abhi6168/ABCD_CustomerAgent_Qwen_2.5_7b bash run.sh
```

Execution order:
1. `scripts/train_numerical.sh`
2. `scripts/train_nlp.sh` (or skipped)
3. `scripts/train_bandit.sh`

---

## 6. Running Individual Approaches

### 6.1 Numerical Multi-Turn RL

PPO agent with pure numeric observations. No LLM required.

```bash
# Default (1M steps)
bash scripts/train_numerical.sh

# Quick smoke-test
NUM_TIMESTEPS=10000 bash scripts/train_numerical.sh

# Production run
NUM_TIMESTEPS=1500000 N_ENVS=4 bash scripts/train_numerical.sh
```

**What it does:**
1. Creates venv, installs dependencies (first run only)
2. Trains PPO with curriculum (easy → medium → full subflows)
3. Exports plots and logs to `output/numerical-multi-turn/`
4. Copies best checkpoint to `best_model/numerical/`

**Output files:**
```
output/numerical-multi-turn/
├── training_log.json           ← step-by-step metrics
├── training_summary.json       ← final results summary
├── full_pipeline_report.json   ← training + eval + baselines
├── demo_rollouts.json          ← example episode traces
├── plots/
│   ├── reward_curve.png
│   ├── resolution_curve.png
│   ├── eval_curve.png
│   ├── mean_reward_leaderboard.png
│   └── terminal_outcomes.png
└── models/
    ├── best_model.zip
    └── final_model.zip
```

---

### 6.2 NLP Multi-Turn RL

PPO + LLM-generated customer utterances.

#### HuggingFace backend (recommended)

```bash
# HF hub download (first run downloads model)
HF_MODEL_PATH=abhi6168/ABCD_CustomerAgent_Qwen_2.5_7b bash scripts/train_nlp.sh

# Local model directory
HF_MODEL_PATH=/path/to/model bash scripts/train_nlp.sh
```

#### Ollama backend

```bash
# Start ollama in a separate terminal
ollama serve && ollama pull llama3

bash scripts/train_nlp.sh
OLLAMA_MODEL=mistral bash scripts/train_nlp.sh
```

Output structure mirrors numerical, under `output/nlp-multi-turn/` and `best_model/nlp/`.

---

### 6.3 Contextual Bandit (stub)

```bash
bash scripts/train_bandit.sh
```

Creates `output/contextual-bandit/` with a status placeholder. Full implementation planned (LinUCB / Thompson Sampling over the same 5-action, 9-feature space).

---

## 7. Output Structure

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
└── contextual-bandit/
    ├── status.txt
    ├── plots/
    ├── logs/
    └── models/

best_model/
├── numerical/
│   ├── best_model.zip
│   └── final_model.zip
└── nlp/
    ├── best_model.zip
    └── final_model.zip
```

---

## 8. Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `NUM_TIMESTEPS` | `1000000` | PPO training steps |
| `EVAL_EPISODES` | `200` | Episodes for policy evaluation |
| `N_ENVS` | `1` | Parallel training environments |
| `SKIP_NLP` | `0` | Set `1` to skip NLP run (used in `run.sh`) |
| `HF_MODEL_PATH` | _(empty)_ | HF hub repo ID or local path; if set, uses HF backend |
| `OLLAMA_MODEL` | `llama3` | Ollama model tag (used when `HF_MODEL_PATH` is empty) |
| `OLLAMA_ENDPOINT` | `http://localhost:11434/v1` | Ollama API base URL |
| `PYTHON_BIN` | `python3` | Python executable |

---

## 9. LLM Backend Setup

### Option A — HuggingFace hub (downloads automatically)

```bash
HF_MODEL_PATH=abhi6168/ABCD_CustomerAgent_Qwen_2.5_7b bash scripts/train_nlp.sh
```

First run downloads ~14 GB. Cached in `~/.cache/huggingface/` on subsequent runs.

The model loads in float32 on CPU (slow) or bfloat16/float16 on GPU (`device_map=auto`).

### Option B — Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull llama3    # ~4 GB

OLLAMA_MODEL=llama3 bash scripts/train_nlp.sh
```

---

## 10. Windows Local Testing

```bash
# Git Bash — override PYTHON_BIN if python3 not available
PYTHON_BIN=python bash scripts/train_numerical.sh

# Conda environment
PYTHON_BIN="C:/ProgramData/miniconda3/envs/bandit/python.exe" bash scripts/train_numerical.sh
```

WSL2 works without modification.

---

## 11. Troubleshooting

### `python3: command not found`
```bash
PYTHON_BIN=python bash scripts/train_numerical.sh
```

### `OMP: Error #15: Initializing libiomp5md.dll`
Already handled via `KMP_DUPLICATE_LIB_OK=TRUE` in `scripts/common.sh`.

### `ModuleNotFoundError: No module named 'simulation'`
`PYTHONPATH` must include `multiturn_rl/`. Set automatically by `common.sh`. For direct Python invocation:
```bash
export PYTHONPATH="$PWD/multiturn_rl"
python multiturn_rl/simulation/scripts/phase10_full_pipeline.py --help
```

### Ollama connection refused
```bash
ollama serve
curl http://localhost:11434/api/tags
```

### Training crashes mid-run

Resume from checkpoint:
```bash
python multiturn_rl/simulation/scripts/phase10_full_pipeline.py \
  --artifacts-root multiturn_rl/simulation/artifacts \
  --output-subdir run_numerical \
  --timesteps 1500000 \
  --continue-from 400000
```

### Plots not generated

```bash
python tools/export_multiturn_results.py \
  --artifacts-root multiturn_rl/simulation/artifacts \
  --run-subdir run_numerical \
  --out-dir output/numerical-multi-turn
```
