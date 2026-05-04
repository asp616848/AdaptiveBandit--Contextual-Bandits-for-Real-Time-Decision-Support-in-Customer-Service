#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
TIMESTEPS="${TIMESTEPS:-150000}"
N_ENVS="${N_ENVS:-1}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-rl_qwen_long}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MODEL_PATH="${SUPPORT_SIM_LOCAL_MODEL_PATH:-Qwen2.5-7B-Instruct-merged}"
CUSTOMER_MODEL="${SUPPORT_SIM_LLM_MODEL:-local-qwen}"
INTENT_MODEL="${SUPPORT_SIM_INTENT_MODEL:-$CUSTOMER_MODEL}"
AGENT_MODEL="${SUPPORT_SIM_AGENT_MODEL:-$CUSTOMER_MODEL}"
METRICS_EVAL_FREQ="${METRICS_EVAL_FREQ:-500}"
HEARTBEAT_FREQ_STEPS="${HEARTBEAT_FREQ_STEPS:-25}"
CHECKPOINT="${CHECKPOINT:-}"

if [[ "$N_ENVS" != "1" ]]; then
  echo "Direct local Qwen training should use N_ENVS=1 to avoid loading the model in multiple workers." >&2
  exit 1
fi

if [[ ! -d "$ROOT_DIR/$MODEL_PATH" && ! -d "$MODEL_PATH" ]]; then
  echo "Local Qwen model folder not found: $MODEL_PATH" >&2
  echo "Expected something like: $ROOT_DIR/Qwen2.5-7B-Instruct-merged" >&2
  exit 1
fi

RUN_DIR="$ROOT_DIR/rl_training_server/runs/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
PID_DIR="$RUN_DIR/pids"
LOG_FILE="$LOG_DIR/train.log"
PID_FILE="$PID_DIR/rl_training.pid"
LATEST_LINK="$ROOT_DIR/rl_training_server/runs/latest"

mkdir -p "$LOG_DIR" "$PID_DIR"
ln -sfn "$RUN_DIR" "$LATEST_LINK"

cat > "$RUN_DIR/config.json" <<JSON
{
  "run_id": "$RUN_ID",
  "timesteps": $TIMESTEPS,
  "n_envs": $N_ENVS,
  "output_subdir": "$OUTPUT_SUBDIR",
  "backend": "local",
  "model_path": "$MODEL_PATH",
  "customer_model": "$CUSTOMER_MODEL",
  "intent_model": "$INTENT_MODEL",
  "agent_model": "$AGENT_MODEL",
  "metrics_eval_freq": $METRICS_EVAL_FREQ,
  "heartbeat_freq_steps": $HEARTBEAT_FREQ_STEPS,
  "checkpoint": "$CHECKPOINT",
  "artifact_run_dir": "Simulation_4/artifacts/$OUTPUT_SUBDIR/runs/$RUN_ID"
}
JSON

export SUPPORT_SIM_LLM_BACKEND=local
export SUPPORT_SIM_LOCAL_MODEL_PATH="$MODEL_PATH"
export SUPPORT_SIM_LLM_MODEL="$CUSTOMER_MODEL"
export SUPPORT_SIM_INTENT_MODEL="$INTENT_MODEL"
export SUPPORT_SIM_AGENT_MODEL="$AGENT_MODEL"
export METRICS_EVAL_FREQ="$METRICS_EVAL_FREQ"
export HEARTBEAT_FREQ_STEPS="$HEARTBEAT_FREQ_STEPS"
export PYTHONUNBUFFERED=1

CMD=(python rl_training_server/scripts/train_rl_qwen.py
  --timesteps "$TIMESTEPS"
  --output-subdir "$OUTPUT_SUBDIR"
  --run-id "$RUN_ID"
  --n-envs "$N_ENVS"
  --model-path "$MODEL_PATH"
  --customer-model "$CUSTOMER_MODEL"
  --intent-model "$INTENT_MODEL"
  --agent-model "$AGENT_MODEL"
  --metrics-eval-freq "$METRICS_EVAL_FREQ"
  --heartbeat-freq-steps "$HEARTBEAT_FREQ_STEPS")

if [[ -n "$CHECKPOINT" ]]; then
  CMD+=(--checkpoint "$CHECKPOINT")
fi

echo "Starting RL training..."
echo "Run dir: $RUN_DIR"
echo "Log: $LOG_FILE"
echo "Command: ${CMD[*]}"

nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "PID=$(cat "$PID_FILE")"
echo "Latest run link: $LATEST_LINK"
echo "Watch logs:"
echo "  tail -f rl_training_server/runs/latest/logs/train.log"
