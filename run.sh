#!/usr/bin/env bash
# ─── run.sh ───────────────────────────────────────────────────────────────────
# End-to-end submission runner — works on a fresh Ubuntu 22.04 install.
#
# Runs all three approaches in order:
#   [1/3] Numerical multi-turn RL  →  output/numerical-multi-turn/
#   [2/3] NLP multi-turn RL        →  output/nlp-multi-turn/
#   [3/3] Contextual bandit        →  output/contextual-bandit/
#
# Idempotent: if an approach already has output, it is skipped.
# Delete the output directory to re-run that approach.
#
# Override any of these at the command line:
#   NUM_TIMESTEPS_NUMERICAL   PPO steps, numerical  (default: 1000000)
#   NUM_TIMESTEPS_NLG         PPO steps, NLP mode   (default: auto — 5000 GPU / 100 CPU)
#   EVAL_EPISODES             eval episodes         (default: 200)
#   N_ENVS                    parallel envs         (default: 1)
#   HF_MODEL                  HF hub repo ID        (default: abhi6168/ABCD_CustomerAgent_Qwen_2.5_7b)
#   PYTHON_BIN                Python executable     (default: python3)
#
# Examples:
#   bash run.sh
#   NUM_TIMESTEPS_NUMERICAL=50000 bash run.sh      # fast smoke-test
#   NUM_TIMESTEPS_NLG=500 bash run.sh              # more NLG steps
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ── 0. System dependencies ─────────────────────────────────────────────────
# Ubuntu 22.04 docker image ships python3 but not python3-venv / python3-pip.
if ! python3 -m venv --help >/dev/null 2>&1 || ! python3 -m pip --version >/dev/null 2>&1; then
  echo "  [setup] Installing python3-venv and python3-pip..."
  apt-get update -qq
  apt-get install -y python3-venv python3-pip
fi

# ── 1. Configuration ───────────────────────────────────────────────────────
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export EVAL_EPISODES="${EVAL_EPISODES:-200}"
export N_ENVS="${N_ENVS:-1}"

# Numerical PPO — state-only, no LLM; fast (~20-30 min on CPU at 1M steps)
_NUM_NUMERICAL="${NUM_TIMESTEPS_NUMERICAL:-1000000}"

# NLP PPO — each env step calls the 7B LLM; much slower per step
# Auto-set based on GPU availability if not overridden.
_NUM_NLG_OVERRIDE="${NUM_TIMESTEPS_NLG:-}"

# HuggingFace model used for NLG customer utterances
_HF_MODEL="abhi6168/ABCD_CustomerAgent_Qwen_2.5_7b"

# ── 2. Venv + base dependencies ────────────────────────────────────────────
# Source common.sh once here to create the venv and install base requirements.
# The individual train scripts source it again — that second pass is fast
# (venv already exists, pip resolves to no-op).
source "$ROOT_DIR/scripts/common.sh"

# ── 3. LLM / HuggingFace dependencies ─────────────────────────────────────
echo "  [setup] Installing LLM dependencies (transformers, accelerate, ...)..."
python -m pip install -r "$ROOT_DIR/requirements-llm.txt" -q

# ── 4. Auto-set NLG timesteps based on hardware ────────────────────────────
if [ -n "$_NUM_NLG_OVERRIDE" ]; then
  _NUM_NLG="$_NUM_NLG_OVERRIDE"
else
  if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    _NUM_NLG=5000
    echo "  [setup] GPU detected — NLG training: ${_NUM_NLG} steps"
  else
    _NUM_NLG=100
    echo "  [setup] No GPU — NLG training: ${_NUM_NLG} steps (CPU-safe demo)"
  fi
fi

# ── 5. Pre-download HuggingFace model ──────────────────────────────────────
echo "  [setup] Checking HF model cache for ${_HF_MODEL}..."
python - <<PYEOF
import os, sys
try:
    from huggingface_hub import snapshot_download, try_to_load_from_cache
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    slug = "models--" + "${_HF_MODEL}".replace("/", "--")
    if os.path.isdir(os.path.join(cache_dir, slug)):
        print("  [setup] HF model already cached — skipping download.")
    else:
        print("  [setup] Downloading HF model (may take 10-30 min depending on connection)...")
        snapshot_download("${_HF_MODEL}")
        print("  [setup] Download complete.")
except Exception as e:
    print(f"  [setup] WARNING: HF model download failed: {e}", file=sys.stderr)
    print("  [setup] NLG run will attempt to download at training time.", file=sys.stderr)
PYEOF

# ── Idempotency helper ─────────────────────────────────────────────────────
_is_done() {
  local summary="$ROOT_DIR/$1/training_summary.json"
  if [ -f "$summary" ]; then
    echo "  [skip] $1 already complete. Delete that folder to re-run."
    return 0
  fi
  return 1
}

# ── Banner ─────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║   AdaptiveBandit — Contextual Bandits for Customer Service RL        ║"
echo "║   End-to-end submission runner                                        ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Numerical : ${_NUM_NUMERICAL} steps"
echo "  NLG       : ${_NUM_NLG} steps  (model: ${_HF_MODEL})"
echo ""

# ── [1/3] Numerical multi-turn RL ─────────────────────────────────────────
echo "  [1/3] Numerical multi-turn RL..."
if ! _is_done "output/numerical-multi-turn"; then
  NUM_TIMESTEPS="$_NUM_NUMERICAL" bash "$ROOT_DIR/scripts/train_numerical.sh"
fi

# ── [2/3] NLP multi-turn RL ───────────────────────────────────────────────
echo ""
echo "  [2/3] NLP multi-turn RL  (HuggingFace backend)..."
if ! _is_done "output/nlp-multi-turn"; then
  NUM_TIMESTEPS="$_NUM_NLG" \
  HF_MODEL_PATH="$_HF_MODEL" \
  bash "$ROOT_DIR/scripts/train_nlp.sh"
fi

# ── [3/3] Contextual bandit ───────────────────────────────────────────────
echo ""
echo "  [3/3] Contextual bandit..."

_CB_DIR="$ROOT_DIR/Contextual Bandits"
_CB_OUT="$ROOT_DIR/output/contextual-bandit"

if ! _is_done "output/contextual-bandit"; then
  if [ -f "$_CB_DIR/main.py" ]; then
    mkdir -p "$_CB_OUT/plots"

    # Install contextual bandit dependencies
    if [ -f "$_CB_DIR/requirements.txt" ]; then
      echo "  [setup] Installing Contextual Bandit dependencies..."
      python -m pip install -r "$_CB_DIR/requirements.txt" -q
    fi

    # Train
    echo "  Running Contextual Bandit training (this may take a bit)..."
    python "$_CB_DIR/main.py" --output-root "$_CB_OUT"

    # Visualize
    if [ -f "$_CB_DIR/plot_bandit_results.py" ]; then
      echo "  Generating Contextual Bandit plots..."
      python "$_CB_DIR/plot_bandit_results.py" --run-dir "$_CB_OUT"
    fi

    # Mark as done
    echo '{"status": "completed"}' > "$_CB_OUT/training_summary.json"
  else
    echo "  [skip] Contextual Bandits/main.py not found."
  fi
fi

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║   All done.                                                           ║"
echo "║     output/numerical-multi-turn/                                      ║"
echo "║     output/nlp-multi-turn/                                            ║"
echo "║     output/contextual-bandit/                                         ║"
echo "║     best_model/numerical/                                             ║"
echo "║     best_model/nlp/                                                   ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""