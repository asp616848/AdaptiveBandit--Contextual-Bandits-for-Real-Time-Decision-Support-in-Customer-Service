#!/usr/bin/env bash
# ─── scripts/common.sh ────────────────────────────────────────────────────────
# Shared setup sourced by every training script.
# Source this file AFTER setting ROOT_DIR to the repo root.
# ─────────────────────────────────────────────────────────────────────────────

# ── Python / venv ──────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="$ROOT_DIR/.venv"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: '$PYTHON_BIN' not found. Set PYTHON_BIN to your Python 3.9+ executable." >&2
  exit 1
fi

if [ ! -f "$VENV_DIR/bin/activate" ] && [ ! -f "$VENV_DIR/Scripts/activate" ]; then
  echo "  Creating virtual environment at $VENV_DIR ..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
if [ -f "$VENV_DIR/bin/activate" ]; then
  source "$VENV_DIR/bin/activate"
else
  source "$VENV_DIR/Scripts/activate"
fi

python -m pip install --upgrade pip setuptools wheel -q
python -m pip install -r "$ROOT_DIR/requirements.txt" -q

# ── Runtime environment ────────────────────────────────────────────────────
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH="$ROOT_DIR/multiturn_rl${PYTHONPATH:+:$PYTHONPATH}"

# ── Configurable training defaults ────────────────────────────────────────
NUM_TIMESTEPS="${NUM_TIMESTEPS:-1000000}"
EVAL_EPISODES="${EVAL_EPISODES:-200}"
N_ENVS="${N_ENVS:-1}"

# ── Fixed paths (relative to ROOT_DIR) ────────────────────────────────────
ARTIFACTS_ROOT="$ROOT_DIR/multiturn_rl/simulation/artifacts"
PIPELINE_SCRIPT="$ROOT_DIR/multiturn_rl/simulation/scripts/phase10_full_pipeline.py"
EXPORT_SCRIPT="$ROOT_DIR/tools/export_multiturn_results.py"

# ── Helper: copy best/final model zips to best_model/<approach>/ ──────────
_copy_best_model() {
  local src_artifacts="$1"
  local src_subdir="$2"
  local dest_dir="$3"

  mkdir -p "$dest_dir"
  for f in best_model.zip final_model.zip; do
    local src="$src_artifacts/$src_subdir/models/$f"
    if [ -f "$src" ]; then
      cp "$src" "$dest_dir/$f"
      echo "    -> Saved $f  →  $dest_dir/"
    fi
  done
}

# ── Helper: print a section banner ────────────────────────────────────────
_banner() {
  echo ""
  echo "══════════════════════════════════════════════════════════════════════"
  echo "  $*"
  echo "══════════════════════════════════════════════════════════════════════"
}
