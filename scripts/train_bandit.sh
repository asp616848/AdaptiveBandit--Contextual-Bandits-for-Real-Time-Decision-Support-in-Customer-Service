#!/usr/bin/env bash
# ─── scripts/train_bandit.sh ──────────────────────────────────────────────────
# Standalone: contextual bandit training.
# NOTE: Full implementation is pending. This script creates the expected output
#       directory structure and a status placeholder so the submission pipeline
#       completes cleanly.
#
# Output:
#   output/contextual-bandit/
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/common.sh"

_banner "Contextual Bandit  (stub)"

mkdir -p output/contextual-bandit/plots
mkdir -p output/contextual-bandit/logs
mkdir -p output/contextual-bandit/models

cat > output/contextual-bandit/status.txt <<'EOF'
Contextual Bandit — implementation pending.

Planned approach:
  - LinUCB / Thompson Sampling over customer support context features
  - Context: sentiment, frustration, info_proxy, turn_count, last_action
  - Arms: the 5 agent actions (Greet, Request Info, Provide Info, Escalate, Close)
  - Reward: same shaped reward as SupportEnv (resolution bonus, frustration penalty)

Output will match the output/numerical-multi-turn/ structure once implemented.
EOF

echo "  Stub placeholder written to output/contextual-bandit/status.txt"

_banner "Contextual bandit step complete (stub)"
echo "  Output: output/contextual-bandit/"
