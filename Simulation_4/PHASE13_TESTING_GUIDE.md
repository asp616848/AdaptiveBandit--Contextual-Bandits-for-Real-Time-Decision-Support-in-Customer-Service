# Phase 13 Model Testing Guide

## Overview

This testing framework validates the Phase 13 PPO model which has **significant architectural changes** compared to Phase 10:

### Key Differences

| Component | Phase 10 | Phase 13 |
|-----------|----------|---------|
| **Observation** | Raw state vector | **9D NLP intent vector** |
| **Source** | StateEngine | **IntentClassifier (LLM)** |
| **Reward Logic** | Base reward only | **Potential-based shaping** |
| **Network** | Generic MLP | **[64, 32] optimized** |

---

## What's Included

### 1. Python Test Script
**Path:** `Simulation_4/scripts/phase13_test.py`

```bash
python Simulation_4/scripts/phase13_test.py [--model PATH]
```

**Tests:**
- TEST 1: Model architecture (9D, Discrete(5), [64,32])
- TEST 2: Reward structure (potential-based shaping)

**Output:** Console + `Simulation_4/artifacts/phase13/test_results.json`

---

### 2. Jupyter Notebook
**Path:** `Simulation_4/subnotebooks/12_phase13_model_testing.ipynb`

```bash
jupyter notebook Simulation_4/subnotebooks/12_phase13_model_testing.ipynb
```

**Features:**
- Interactive cell-by-cell testing
- Detailed explanations
- Inline visualization
- Easy to modify

---

## TEST DESCRIPTIONS

### TEST 1: Model Architecture Validation

**What it checks:**
- ✓ Model loads successfully
- ✓ Observation space is 9D (not some other dimension)
- ✓ Action space is Discrete(5)
- ✓ Network is MLP

**Why it matters:**
Phase 13 uses a 9D NLP observation space derived from IntentClassifier. If this dimension is wrong, inference fails.

**Expected output:**
```
TEST 1: Model Architecture Validation
=====================================
✓ Model loaded successfully
  Observation space: Box([9,], float32) → Size: 9 (expected 9)
  ✓ Observation space matches Phase 13 spec (9D)
  Action space: Discrete(5) → Size: 5 (expected 5)
  ✓ Action space matches Phase 13 spec
  => TEST 1: PASSED ✓
```

---

### TEST 2: Reward Structure Validation

**What it checks:**
- ✓ RewardShaper instantiated correctly
- ✓ Strict potential mode enabled (policy-invariant)
- ✓ Potential function computes correctly
- ✓ Reward shaping applies: `reward_new = base + γ·Φ(s') - Φ(s)`

**Why it matters:**
Phase 13 uses **potential-based shaping** to speed up learning without changing optimal policy. Incorrect shaping leads to suboptimal policies.

**Expected output:**
```
TEST 2: Reward Structure Validation
===================================
✓ RewardShaper instantiated
  Enabled: True
  Strict potential mode: True
  Info gain bonus: 0.05
  Progress increase bonus: 0.10
  Gamma: 0.99

  Potential function test:
    Pre-state potential: 0.0355
    Post-state potential: 0.1150
    Shaped reward (base=1.0): 1.2594
  => TEST 2: PASSED ✓
```

---

## Usage Guide

### Quick Validation (30 seconds)

```bash
cd Simulation_4
python scripts/phase13_test.py
```

This quickly validates the model architecture without expensive computations.

---

### Interactive Exploration (5 minutes)

```bash
jupyter notebook subnotebooks/12_phase13_model_testing.ipynb
```

Run cells 1-3 to understand each component with detailed output.

---

### Custom Model Path

```bash
python Simulation_4/scripts/phase13_test.py \
  --model Simulation_4/artifacts/phase13/models/best_model.zip
```

---

## Interpreting Results

### ✓ All Tests Pass

**Interpretation:** Phase 13 model is correctly implemented with:
- Proper 9D observation space
- Correct reward shaping formula
- Expected network architecture

**Next steps:**
- Run batch evaluation on real data
- Compare metrics with Phase 10
- Deploy if superior

---

### ✗ TEST 1 Fails (Architecture)

**Problem:** Observation or action space mismatch

**Possible causes:**
- Wrong model file (using Phase 10 instead of Phase 13)
- Phase 13 model not trained yet
- Corrupted checkpoint

**Solution:**
```bash
# Check model exists
ls -la Simulation_4/artifacts/phase13/models/best_model.zip

# Or use Phase 10 for comparison
python Simulation_4/scripts/phase13_test.py \
  --model Simulation_4/artifacts/phase10_prod/models/best_model.zip
```

---

### ✗ TEST 2 Fails (Reward Structure)

**Problem:** RewardShaper not configured correctly

**Possible causes:**
- Imports fail (missing training module)
- reward_shaping.py modified

**Solution:**
```bash
# Check imports work
python -c "from Simulation_4.training.reward_shaping import RewardShaper; print('OK')"

# Check file exists
ls -la Simulation_4/training/reward_shaping.py
```

---

## Performance Targets

Phase 13 with proper training should achieve:
- **Model architecture:** ✓ 9D obs, [64,32] network
- **Reward shaping:** ✓ Potential-based enabled
- **Mean reward:** ~2.0+ (better than Phase 10)
- **Resolution rate:** ~60%+
- **Dropout rate:** <15%

---

## Advanced: Understanding the Changes

### Why 9D NLP Observation?

**Phase 10:** Raw state from StateEngine
```
[turn_count, information, progress, frustration, ...]
```

**Phase 13:** Semantic features from IntentClassifier
```
1. intent_norm (which intent does customer have?)
2. confidence (how confident is classifier?)
3. sentiment_norm (customer's mood)
4. suggested_action_norm (what should agent do?)
5. escalation_flag (escalate needed?)
6. info_completeness (enough info gathered?)
7. turn_count_norm (conversation length)
8. history_depth_norm (conversation complexity)
9. customer_len_norm (customer message length)
```

This is **richer, more semantic** than raw numbers.

---

### Why Potential-Based Shaping?

**Problem:** Raw reward can be noisy and sparse
**Solution:** Add potential-based shaping

```
new_reward = base_reward + γ·Φ(s') - Φ(s)

Where Φ(s) = information advantages
           + progress advantages  
           - frustration penalties
```

This is **policy-invariant** (doesn't change optimal policy) but **accelerates learning**.

---

## Troubleshooting

### ImportError: "No module named Simulation_4"

```bash
# Make sure you're in repo root
cd ~/RL_Project/AdaptiveBandit--Contextual-Bandits-for-Real-Time-Decision-Support-in-Customer-Service

# Or add path
export PYTHONPATH="$(pwd):$PYTHONPATH"
python Simulation_4/scripts/phase13_test.py
```

---

### "Model loading timeout"

LLM (phi3/llama3) is being loaded from HuggingFace

```bash
# This is fine, tests skip LLM initialization
# Just takes longer on first run
```

---

### "No such file or directory"

Check paths are correct:

```bash
# Verify architecture
pwd  # Should be repo root
ls Simulation_4/scripts/phase13_test.py  # Should exist
ls Simulation_4/artifacts/phase10_prod/models/best_model.zip  # Model should exist
```

---

## Files Reference

```
Simulation_4/
├── scripts/
│   └── phase13_test.py              ← Main test script
├── subnotebooks/
│   └── 12_phase13_model_testing.ipynb  ← Interactive notebook
├── PHASE13_QUICK_REFERENCE.md       ← Quick commands
├── PHASE13_TESTING_GUIDE.md         ← This file
└── PHASE13_TRAINING_LOGIC_ACADEMIC.md  ← Training details
```

---

## Summary

✅ **Created:** Phase 13 testing framework  
✅ **Tests:** Architecture + Reward structure  
✅ **Tools:** Python script + Jupyter notebook  
✅ **Docs:** Quick reference + Detailed guide  
✅ **Duration:** 30 seconds to 5 minutes  

**Next:** Run `python Simulation_4/scripts/phase13_test.py` or open the notebook

---

**Last updated:** April 27, 2026
