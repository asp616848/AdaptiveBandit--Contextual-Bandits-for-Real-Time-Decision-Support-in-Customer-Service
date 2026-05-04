#!/usr/bin/env bash
# ─── scripts/train_numerical.sh ───────────────────────────────────────────────
# Standalone: train PPO agent with state-only (non-NLG) observations.
#
# Configurable env vars (set before running):
#   NUM_TIMESTEPS   PPO training steps          (default: 1000000)
#   EVAL_EPISODES   episodes for final eval     (default: 200)
#   N_ENVS          parallel training envs      (default: 1)
#   PYTHON_BIN      Python executable           (default: python3)
#
# Output:
#   output/numerical-multi-turn/   plots, logs, model zips
#   best_model/numerical/          best_model.zip + final_model.zip
#
# Example:
#   bash scripts/train_numerical.sh
#   NUM_TIMESTEPS=500000 bash scripts/train_numerical.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/common.sh"

_banner "Numerical multi-turn RL  (PPO · state-only observations · ${NUM_TIMESTEPS} steps)"

mkdir -p output/numerical-multi-turn

python "$PIPELINE_SCRIPT" \
  --artifacts-root "$ARTIFACTS_ROOT" \
  --output-subdir  run_numerical \
  --timesteps      "$NUM_TIMESTEPS" \
  --eval-episodes  "$EVAL_EPISODES" \
  --n-envs         "$N_ENVS" \
  --continue-from  _none_ \
  --skip-validation

python "$EXPORT_SCRIPT" \
  --artifacts-root "$ARTIFACTS_ROOT" \
  --run-subdir     run_numerical \
  --out-dir        output/numerical-multi-turn

_copy_best_model "$ARTIFACTS_ROOT" run_numerical "$ROOT_DIR/best_model/numerical"

_banner "Numerical run complete"
echo "  Plots + logs : output/numerical-multi-turn/"
echo "  Best model   : best_model/numerical/"
