#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# -----------------------------
# 1) Python venv + dependencies
# -----------------------------
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python3 is required but was not found." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# Offline/CI-friendly defaults (avoid any model downloads)
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

# -----------------------------
# 2) Output directories
# -----------------------------
mkdir -p output/nlp-multi-turn output/numerical-multi-turn

ARTIFACTS_ROOT="Multi-Turn RL/Simulation_4/artifacts"
PIPELINE_SCRIPT="Multi-Turn RL/Simulation_4/scripts/phase10_full_pipeline.py"

# Faster defaults for docker evaluation (override by exporting env vars)
NUM_TIMESTEPS="${NUM_TIMESTEPS:-20000}"
NLP_TIMESTEPS="${NLP_TIMESTEPS:-20000}"
EVAL_EPISODES="${EVAL_EPISODES:-200}"
DEMO_EPISODES="${DEMO_EPISODES:-6}"
N_ENVS="${N_ENVS:-1}"

# -----------------------------
# 3) Numerical/state-only run
# -----------------------------
echo "=== [1/2] Multi-turn numerical (state observation) ==="
python "$PIPELINE_SCRIPT" \
  --artifacts-root "$ARTIFACTS_ROOT" \
  --output-subdir "run_numerical" \
  --timesteps "$NUM_TIMESTEPS" \
  --eval-episodes "$EVAL_EPISODES" \
  --demo-episodes "$DEMO_EPISODES" \
  --n-envs "$N_ENVS" \
  --continue-from "_none_" \
  --skip-validation

python tools/export_multiturn_results.py \
  --artifacts-root "$ARTIFACTS_ROOT" \
  --run-subdir "run_numerical" \
  --out-dir "output/numerical-multi-turn"

# -----------------------------
# 4) NLP/text-observation run
# -----------------------------
echo "=== [2/2] Multi-turn NLP (text-only observation; offline-safe) ==="
python "$PIPELINE_SCRIPT" \
  --artifacts-root "$ARTIFACTS_ROOT" \
  --output-subdir "run_nlp" \
  --timesteps "$NLP_TIMESTEPS" \
  --eval-episodes "$EVAL_EPISODES" \
  --demo-episodes "$DEMO_EPISODES" \
  --n-envs "$N_ENVS" \
  --continue-from "_none_" \
  --skip-validation \
  --text-only-observation

python tools/export_multiturn_results.py \
  --artifacts-root "$ARTIFACTS_ROOT" \
  --run-subdir "run_nlp" \
  --out-dir "output/nlp-multi-turn"

echo "\nDone. Outputs written under ./output/"
