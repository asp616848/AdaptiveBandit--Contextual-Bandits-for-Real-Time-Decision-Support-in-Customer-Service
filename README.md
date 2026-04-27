# AdaptiveBandit — RL for Real‑Time Decision Support in Customer Service

This repository contains a **multi-turn customer support simulator** and PPO training/evaluation pipelines.

Demo video: (add link)

## One-command evaluation (required)

The project is evaluated **only** by running the bash script:

```bash
bash run.sh
```

`run.sh` will:
- Create and activate a Python virtual environment
- Install dependencies
- Run the full pipeline end-to-end (training + evaluation)
- Save all outputs (reports + plots + models) under `output/`

## Output structure

After `run.sh`, you should see:

```
output/
	numerical/
		*.json
		models/
		plots/
	text/
		*.json
		models/
		plots/
```

## Repository structure

- `Multi-Turn RL/Simulation_4/`: simulator, PPO training code, and pipeline scripts
- `Multi-Turn RL/Qwen2.5-7B-Instruct-merged/`: local fine-tuned model snapshot (not required for Docker evaluation)
- `tools/`: small utilities used by `run.sh` to export plots/tables

## Notes

- The Docker evaluator runs in `ubuntu:22.04` on a fresh clone.
- `run.sh` is configured to run **offline-safe** (no external model downloads). The “NLP” run uses a deterministic hashing text embedding fallback.

## Optional: full LLM NLP (Phase 13)

The graded evaluation does **not** require LLMs, but the repo also includes an optional Phase 13 “full NLP” pipeline.

- Install optional deps: `pip install -r requirements-llm.txt`
- Run via unified helper:
	- `python "Multi-Turn RL/run_multiturn.py" llm --timesteps 500000 --intent-model phi3 --n-envs 1`

### Backend toggle (Ollama vs HF)

`IntentClassifier` and `AgentResponseGenerator` can route to different backends via env vars:

- Ollama (OpenAI-compatible):
	- `SUPPORT_SIM_LLM_BACKEND=ollama`
	- `SUPPORT_SIM_LLM_ENDPOINT=http://localhost:11434/v1`
- Hugging Face (local or remote):
	- `SUPPORT_SIM_LLM_BACKEND=hf-qwen`
	- `SUPPORT_SIM_HF_MODEL_ID=abhi6168/ABCD_CustomerAgent_Qwen_2.5_7b`
	- (optional) `SUPPORT_SIM_HF_MODEL_PATH=Multi-Turn RL/Qwen2.5-7B-Instruct-merged`

To download the model snapshot locally:

`python tools/download_qwen_model.py --repo-id abhi6168/ABCD_CustomerAgent_Qwen_2.5_7b --dst "Multi-Turn RL/Qwen2.5-7B-Instruct-merged"`