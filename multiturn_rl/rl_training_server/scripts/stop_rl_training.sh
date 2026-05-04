#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/rl_training_server/runs/latest}"
PID_FILE="$RUN_DIR/pids/rl_training.pid"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  kill "$(cat "$PID_FILE")"
  echo "Stopped RL training PID=$(cat "$PID_FILE")"
else
  echo "No running training PID found."
fi

