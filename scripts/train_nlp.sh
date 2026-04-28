#!/usr/bin/env bash
# ─── scripts/train_nlp.sh ─────────────────────────────────────────────────────
# Standalone: train PPO agent with NLG-enabled customer utterances via LLM.
# Supports two LLM backends:
#   ollama  — default; requires ollama running locally or remotely
#   hf      — HuggingFace model (local path or HF hub repo ID)
#
# Configurable env vars:
#   NUM_TIMESTEPS   PPO training steps          (default: 1000000)
#   EVAL_EPISODES   episodes for final eval     (default: 200)
#   N_ENVS          parallel training envs      (default: 1)
#   HF_MODEL_PATH   HF hub repo ID or local path (e.g. abhi6168/ABCD_CustomerAgent_Qwen_2.5_7b)
#                   If set, uses HF backend. If empty, uses ollama.
#   OLLAMA_MODEL    ollama model tag            (default: llama3)
#   OLLAMA_ENDPOINT ollama base URL             (default: http://localhost:11434/v1)
#   PYTHON_BIN      Python executable           (default: python3)
#
# Examples:
#   # HuggingFace hub model (downloads on first run)
#   HF_MODEL_PATH=abhi6168/ABCD_CustomerAgent_Qwen_2.5_7b bash scripts/train_nlp.sh
#
#   # local model directory
#   HF_MODEL_PATH=/path/to/model bash scripts/train_nlp.sh
#
#   # ollama backend
#   OLLAMA_MODEL=llama3 bash scripts/train_nlp.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/common.sh"

# ── NLP-specific defaults ──────────────────────────────────────────────────
HF_MODEL_PATH="${HF_MODEL_PATH:-}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3}"
OLLAMA_ENDPOINT="${OLLAMA_ENDPOINT:-http://localhost:11434/v1}"

# ── Resolve LLM backend ────────────────────────────────────────────────────
if [ -n "$HF_MODEL_PATH" ]; then
  _llm_backend="hf"
  _llm_extra="--hf-model-path $HF_MODEL_PATH"
  _backend_label="HuggingFace ($HF_MODEL_PATH)"

  # Install LLM dependencies (transformers, accelerate, etc.)
  echo "  Installing LLM dependencies ..."
  python -m pip install -r "$ROOT_DIR/requirements-llm.txt" -q
else
  _llm_backend="ollama"
  _llm_extra="--ollama-model $OLLAMA_MODEL"
  _backend_label="ollama ($OLLAMA_MODEL @ $OLLAMA_ENDPOINT)"

  # Verify ollama is reachable before starting a long training run
  _tags_url="${OLLAMA_ENDPOINT%/v1}/api/tags"
  if ! python -c "import requests; r=requests.get('$_tags_url',timeout=5); exit(0 if r.status_code==200 else 1)" 2>/dev/null; then
    echo ""
    echo "ERROR: Ollama not reachable at $OLLAMA_ENDPOINT" >&2
    echo "  Start ollama:       ollama serve" >&2
    echo "  Pull a model:       ollama pull $OLLAMA_MODEL" >&2
    echo "  Or use HF backend:  HF_MODEL_PATH=<repo-id-or-path> bash scripts/train_nlp.sh" >&2
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
