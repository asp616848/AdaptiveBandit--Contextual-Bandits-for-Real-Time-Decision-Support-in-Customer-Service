#!/usr/bin/env bash
# ─── run.sh ───────────────────────────────────────────────────────────────────
# End-to-end submission script.
# Runs all three approaches in sequence:
#   [1/3] Numerical multi-turn RL  →  output/numerical-multi-turn/
#   [2/3] NLP multi-turn RL        →  output/nlp-multi-turn/        (auto-skipped if no LLM)
#   [3/3] Contextual bandit        →  output/contextual-bandit/     (stub)
#
# Each step delegates to its own script under scripts/. They can also be run
# individually — see COMMANDS.md for one-liners.
#
# Configurable env vars (export before calling):
#   NUM_TIMESTEPS   PPO training steps          (default: 1000000)
#   EVAL_EPISODES   eval episodes per run       (default: 200)
#   N_ENVS          parallel training envs      (default: 1)
#   SKIP_NLP        set 1 to skip NLP run       (default: 0)
#   OLLAMA_MODEL    ollama model name           (default: llama3)
#   OLLAMA_ENDPOINT ollama base URL             (default: http://localhost:11434/v1)
#   HF_MODEL_PATH   path to local HF model dir (default: "", uses ollama)
#   PYTHON_BIN      Python executable           (default: python3)
#
# Ubuntu 22.04 quick start:
#   bash run.sh                          # full run (NLP skipped if ollama absent)
#   SKIP_NLP=1 bash run.sh               # skip NLP
#   NUM_TIMESTEPS=50000 SKIP_NLP=1 bash run.sh   # fast smoke-test
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ── Banner ─────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║   AdaptiveBandit — Contextual Bandits for Customer Service RL        ║"
echo "║   End-to-end submission runner                                        ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# ── [1/3] Numerical multi-turn ────────────────────────────────────────────
echo "  [1/3] Starting: Numerical multi-turn RL ..."
bash "$ROOT_DIR/scripts/train_numerical.sh"

# ── [2/3] NLP multi-turn ──────────────────────────────────────────────────
echo ""
echo "  [2/3] Starting: NLP multi-turn RL ..."

_skip_nlp="${SKIP_NLP:-0}"

if [ "$_skip_nlp" = "1" ]; then
  echo "        Skipped (SKIP_NLP=1)."
else
  # Auto-detect: if HF_MODEL_PATH not set, probe ollama; skip gracefully if absent
  _hf_path="${HF_MODEL_PATH:-}"
  if [ -z "$_hf_path" ]; then
    _ollama_ep="${OLLAMA_ENDPOINT:-http://localhost:11434/v1}"
    _tags_url="${_ollama_ep%/v1}/api/tags"
    # Use the venv python (already activated by train_numerical.sh's sourcing of common.sh)
    if ! python -c "import requests; r=requests.get('$_tags_url',timeout=3); exit(0 if r.status_code==200 else 1)" 2>/dev/null; then
      echo "        Ollama not reachable at $_ollama_ep — NLP run skipped."
      echo "        To enable: start ollama, or set HF_MODEL_PATH to a local model."
      _skip_nlp=1
    fi
  fi
fi

if [ "$_skip_nlp" != "1" ]; then
  bash "$ROOT_DIR/scripts/train_nlp.sh"
fi

# ── [3/3] Contextual bandit ───────────────────────────────────────────────
echo ""
echo "  [3/3] Starting: Contextual bandit ..."
bash "$ROOT_DIR/scripts/train_bandit.sh"

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║   All done. Outputs:                                                  ║"
echo "║     output/numerical-multi-turn/                                      ║"
[ "$_skip_nlp" != "1" ] && \
echo "║     output/nlp-multi-turn/                                            ║"
echo "║     output/contextual-bandit/                                         ║"
echo "║     best_model/numerical/                                             ║"
[ "$_skip_nlp" != "1" ] && \
echo "║     best_model/nlp/                                                   ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
