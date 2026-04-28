#!/usr/bin/env bash
# ─── scripts/train_nlp.sh ─────────────────────────────────────────────────────
# Standalone: train PPO agent with NLG-enabled customer utterances via LLM.
# Supports two LLM backends:
#   ollama  — default; requires ollama running locally or remotely
#   hf      — local HuggingFace model (set HF_MODEL_PATH)
#
# Configurable env vars (set before running):
#   NUM_TIMESTEPS   PPO training steps          (default: 1000000)
#   EVAL_EPISODES   episodes for final eval     (default: 200)
#   N_ENVS          parallel training envs      (default: 1)
#   OLLAMA_MODEL    ollama model name           (default: llama3)
#   OLLAMA_ENDPOINT ollama base URL             (default: http://localhost:11434/v1)
#   HF_MODEL_PATH   path to local HF model dir (default: "", uses ollama)
#   PYTHON_BIN      Python executable           (default: python3)
#
# Output:
#   output/nlp-multi-turn/    plots, logs, model zips
#   best_model/nlp/           best_model.zip + final_model.zip
#
# Examples:
#   # ollama backend (llama3 must be pulled)
#   bash scripts/train_nlp.sh
#
#   # custom ollama model
#   OLLAMA_MODEL=mistral bash scripts/train_nlp.sh
#
#   # local HuggingFace model
#   HF_MODEL_PATH="Multi-Turn RL/Qwen2.5-7B-Instruct-merged" bash scripts/train_nlp.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/common.sh"

# ── Resolve LLM backend ────────────────────────────────────────────────────
if [ -n "$HF_MODEL_PATH" ]; then
  _llm_backend="hf"
  _llm_extra="--hf-model-path $HF_MODEL_PATH"
  _backend_label="HuggingFace ($HF_MODEL_PATH)"
else
  _llm_backend="ollama"
  _llm_extra="--ollama-model $OLLAMA_MODEL"
  _backend_label="ollama ($OLLAMA_MODEL @ $OLLAMA_ENDPOINT)"

  # Verify ollama is reachable before starting a long training run
  _tags_url="${OLLAMA_ENDPOINT%/v1}/api/tags"
  if ! python -c "import requests; r=requests.get('$_tags_url',timeout=5); exit(0 if r.status_code==200 else 1)" 2>/dev/null; then
    echo ""
    echo "ERROR: Ollama not reachable at $OLLAMA_ENDPOINT" >&2
    echo "  Start ollama:            ollama serve" >&2
    echo "  Pull the model:          ollama pull $OLLAMA_MODEL" >&2
    echo "  Or use a local HF model: HF_MODEL_PATH=<path> bash scripts/train_nlp.sh" >&2
    exit 1
  fi
fi

_banner "NLP multi-turn RL  (PPO · NLG enabled · ${NUM_TIMESTEPS} steps · $_backend_label)"

mkdir -p output/nlp-multi-turn

# shellcheck disable=SC2086
python "$PIPELINE_SCRIPT" \
  --artifacts-root "$ARTIFACTS_ROOT" \
  --output-subdir  run_nlp \
  --timesteps      "$NUM_TIMESTEPS" \
  --eval-episodes  "$EVAL_EPISODES" \
  --n-envs         "$N_ENVS" \
  --continue-from  _none_ \
  --skip-validation \
  --nlg-enabled \
  --llm-backend    "$_llm_backend" \
  $_llm_extra

python "$EXPORT_SCRIPT" \
  --artifacts-root "$ARTIFACTS_ROOT" \
  --run-subdir     run_nlp \
  --out-dir        output/nlp-multi-turn

_copy_best_model "$ARTIFACTS_ROOT" run_nlp "$ROOT_DIR/best_model/nlp"

_banner "NLP run complete"
echo "  Plots + logs : output/nlp-multi-turn/"
echo "  Best model   : best_model/nlp/"
