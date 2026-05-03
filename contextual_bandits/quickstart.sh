#!/bin/bash
# Quick start guide for Per-Turn LinUCB training

set -e

cd "/Users/tishabhavsar/RL_Project /AdaptiveBandit--Contextual-Bandits-for-Real-Time-Decision-Support-in-Customer-Service"

echo "==============================================="
echo "Per-Turn LinUCB: Quick Start"
echo "==============================================="

# Step 1: Train LinUCB
echo ""
echo "Step 1: Training Per-Turn LinUCB models..."
echo "  Command: python -m Simulation_4.contextual_bandits.train_linucb"
echo ""
python -m Simulation_4.contextual_bandits.train_linucb \
  --collect-episodes 300 \
  --eval-episodes 200 \
  --alpha 1.0

# Step 2: Compare with PPO
echo ""
echo "Step 2: Comparing LinUCB vs PPO..."
echo "  Command: python -m Simulation_4.contextual_bandits.evaluate_linucb"
echo ""
python -m Simulation_4.contextual_bandits.evaluate_linucb \
  --episodes 150 \
  --output Simulation_4/artifacts/linucb_vs_ppo_final.json

echo ""
echo "==============================================="
echo "✓ Complete! Results saved to:"
echo "  - Simulation_4/artifacts/linucb_models/"
echo "  - Simulation_4/artifacts/linucb_vs_ppo_final.json"
echo "==============================================="
