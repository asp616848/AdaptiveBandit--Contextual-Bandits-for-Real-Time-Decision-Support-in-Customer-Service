# Long PPO Training With Fine-Tuned Qwen + Simulation_4

This directory contains the pipeline to run the **long-running PPO training** on the GPU server. It seamlessly integrates:
  + The `Simulation_4` environment
  + The fine-tuned Hugging Face Qwen customer model (`abhi6168/ABCD_CustomerAgent_Qwen_2.5_7b`)
  + The PPO agent with specific observation masking

## Overview

There is no Ollama, no vLLM, no localhost endpoint, and no web server.

The training process loads the fine-tuned model directly from the Hugging Face Hub (or a downloaded cache) with `transformers` and runs the `Simulation_4` simulator + local Qwen customer model + PPO agent entirely in a single Python process. This approach is highly efficient for single-GPU setups without external networking dependencies.

## 1. Prerequisites

You only need the `multiturn_rl` directory and its required dependencies.

```bash
# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r multiturn_rl/requirements.txt
pip install -r multiturn_rl/Fine_Tune/requirements-server.txt
```

## 2. Smoke Test The Local Qwen Model

This script loads `abhi6168/ABCD_CustomerAgent_Qwen_2.5_7b` directly with `transformers` and issues 3 test prompts (ask_info, failed_solution, escalation_pressure).

```bash
python multiturn_rl/rl_training_server/scripts/smoke_local_qwen.py
```

It should print 3 JSON lines showing that it correctly generated `<behavior>...</behavior>` and parsed the dictionary. The results are also saved to `multiturn_rl/rl_training_server/runs/smoke/local_qwen_behavior_*.jsonl`.

## 3. Smoke Test Simulator + Local Qwen + NLP Observation

This script fully instantiates `SupportEnv`, wraps it in `NLPObservationWrapper` (which boots the Qwen model), applies the action masking, and steps through 6 dummy actions.

```bash
python multiturn_rl/rl_training_server/scripts/smoke_simulator_nlp.py
```

It should print a JSON dictionary with the steps. If it successfully finishes without crashing, the full stack is working.

## 4. Start Full RL Training

A helper script wraps the background process launch.

```bash
TIMESTEPS=150000 N_ENVS=1 bash multiturn_rl/rl_training_server/scripts/start_rl_training.sh
```

**Important**: keep `N_ENVS=1`. Direct local Qwen loading with multiple workers can load the model multiple times and exhaust GPU memory.

By default, logs and metrics are written to `multiturn_rl/rl_training_server/runs/<timestamp>`. You can tail the log file as suggested in the script output.

If you ever need to point to a local directory instead of the Hub ID, you can override the path:
```bash
SUPPORT_SIM_LOCAL_MODEL_PATH=/path/to/my/model \
TIMESTEPS=150000 \
bash multiturn_rl/rl_training_server/scripts/start_rl_training.sh
```
