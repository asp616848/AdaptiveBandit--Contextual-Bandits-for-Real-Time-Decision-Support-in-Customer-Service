#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/rl_training_server/runs/latest}"
PID_FILE="$RUN_DIR/pids/rl_training.pid"
LOG_FILE="$RUN_DIR/logs/train.log"
CONFIG_FILE="$RUN_DIR/config.json"

echo "Run dir: $RUN_DIR"
if [[ -f "$CONFIG_FILE" ]]; then
  echo
  echo "Config:"
  cat "$CONFIG_FILE"
  echo
fi

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Training running: PID=$(cat "$PID_FILE")"
  ps -fp "$(cat "$PID_FILE")" || true
else
  echo "Training is not running."
fi

echo
echo "GPU:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true

echo
echo "Heartbeat:"
HEARTBEAT="$(python - <<'PY'
import json
from pathlib import Path
cfg = Path("rl_training_server/runs/latest/config.json")
if cfg.exists():
    data = json.loads(cfg.read_text())
    p = Path(data["artifact_run_dir"]) / "heartbeat.json"
    print(p if p.exists() else "")
PY
)"
if [[ -n "$HEARTBEAT" ]]; then
  cat "$HEARTBEAT"
else
  echo "heartbeat.json not written yet."
fi

echo
echo "Latest training log:"
if [[ -f "$LOG_FILE" ]]; then
  tail -n 120 "$LOG_FILE"
else
  echo "No log found at $LOG_FILE"
fi

echo
echo "Latest metrics:"
METRICS="$(python - <<'PY'
import json
from pathlib import Path
cfg = Path("rl_training_server/runs/latest/config.json")
if cfg.exists():
    data = json.loads(cfg.read_text())
    p = Path(data["artifact_run_dir"]) / "metrics.csv"
    print(p if p.exists() else "")
PY
)"
if [[ -n "$METRICS" ]]; then
  tail -n 10 "$METRICS"
else
  echo "metrics.csv not written yet."
fi
