#!/bin/bash
# ─── VALIDATION_TEST.sh ───────────────────────────────────────────────────────
# Run this script to validate the entire pipeline on a fresh system
# Usage: bash VALIDATION_TEST.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║   AdaptiveBandit — Complete Pipeline Validation Test                 ║"
echo "║   This tests minimal execution: 2 timesteps each method              ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Run with minimal timesteps
echo "[1/3] Testing Numerical Multi-turn RL (2 timesteps)..."
NUM_TIMESTEPS_NUMERICAL=2 bash scripts/train_numerical.sh
echo "✓ Numerical training complete"
echo ""

echo "[2/3] Testing NLP Multi-turn RL (2 timesteps with HuggingFace backend)..."
NUM_TIMESTEPS_NLG=2 bash scripts/train_nlp.sh
echo "✓ NLP training complete"
echo ""

echo "[3/3] Testing Contextual Bandit..."
cd "Contextual Bandits"
python main.py --output-root ../output/contextual-bandit --collect-episodes 2 --eval-episodes 2 --seeds 1
cd ..
echo "✓ Contextual bandit complete"
echo ""

# Validate all outputs exist
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║   Validation Results                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

validate_file() {
    if [ -f "$1" ]; then
        echo "  ✓ $1"
        return 0
    else
        echo "  ✗ MISSING: $1"
        return 1
    fi
}

validate_dir() {
    if [ -d "$1" ]; then
        echo "  ✓ $1/ (exists)"
        return 0
    else
        echo "  ✗ MISSING DIRECTORY: $1/"
        return 1
    fi
}

all_pass=true

echo "Numerical Outputs:"
validate_file "output/numerical-multi-turn/training_summary.json" || all_pass=false
validate_dir "output/numerical-multi-turn/models" || all_pass=false
validate_dir "output/numerical-multi-turn/plots" || all_pass=false

echo ""
echo "NLP Outputs:"
validate_file "output/nlp-multi-turn/training_summary.json" || all_pass=false
validate_dir "output/nlp-multi-turn/models" || all_pass=false
validate_dir "output/nlp-multi-turn/plots" || all_pass=false

echo ""
echo "Contextual Bandit Outputs:"
validate_file "output/contextual-bandit/summary.csv" || all_pass=false
validate_file "output/contextual-bandit/aggregate.csv" || all_pass=false
validate_dir "output/contextual-bandit/models" || all_pass=false
validate_dir "output/contextual-bandit/plots" || all_pass=false

echo ""
echo "Best Models:"
validate_dir "best_model/numerical" || all_pass=false
validate_dir "best_model/nlp" || all_pass=false

echo ""
if [ "$all_pass" = true ]; then
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║   ✓ ALL VALIDATIONS PASSED                                            ║"
    echo "║   The pipeline works end-to-end from a fresh system!                  ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    exit 0
else
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║   ✗ SOME VALIDATIONS FAILED                                           ║"
    echo "║   Check the output above for details                                  ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    exit 1
fi
