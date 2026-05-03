# Phase 13 Testing Quick Reference

## 🚀 Quick Start (Choose One)

### Option A: Run Tests via Command Line
```bash
python Simulation_4/scripts/phase13_test.py
```
Results: Quick validation (30 seconds)

---

### Option B: Interactive Jupyter Notebook (Recommended)
```bash
jupyter notebook Simulation_4/subnotebooks/12_phase13_model_testing.ipynb
```
Run cells sequentially to see detailed output.

---

## 📋 What Gets Tested

| Test | What | Pass When |
|------|------|-----------|
| **1** | Model architecture | 9D obs, Discrete(5) action space |
| **2** | Reward structure | Potential-based shaping enabled |

---

## ✅ Expected Results (All Tests Pass)

```
TEST 1 - Model Architecture      ✓ PASSED
  ✓ 9D observation space (Phase 13 NLP)
  ✓ Discrete(5) action space

TEST 2 - Reward Structure        ✓ PASSED
  ✓ Potential-based shaping enabled
  ✓ Gamma = 0.99 discount factor

🎉 ALL TESTS PASSED
```

---

## ❌ Troubleshooting

### "Model not found"
```bash
ls -la Simulation_4/artifacts/phase10_prod/models/best_model.zip
# If missing, check artifact path
```

### "ImportError: No module named Simulation_4"
```bash
# Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python Simulation_4/scripts/phase13_test.py
```

### "Test 2 fails (Reward structure)"
```bash
# Check RewardShaper is installed
python -c "from Simulation_4.training.reward_shaping import RewardShaper; print('✓ RewardShaper OK')"
```

---

## 🔍 Key Phase 13 Changes vs Phase 10

### Observation Space
- **Phase 10:** Raw state (unknown dimensions)
- **Phase 13:** 9D NLP vector from IntentClassifier ← **TEST 1 validates**

### Reward Structure
- **Phase 10:** Base rewards only
- **Phase 13:** Potential-based shaping: `reward = base + γ·Φ(s') - Φ(s)` ← **TEST 2 validates**

### Network
- **Both:** MLP
- **Phase 13:** Optimized [64, 32] for 9D input

---

## 📊 Quick Commands

```bash
# Test everything
python Simulation_4/scripts/phase13_test.py

# Test with custom model
python Simulation_4/scripts/phase13_test.py --model path/to/model.zip

# Open Jupyter notebook
jupyter notebook Simulation_4/subnotebooks/12_phase13_model_testing.ipynb
```

---

## ⏱️ Test Duration

- **Python script:** ~30 seconds
- **Jupyter notebook:** ~5 minutes (with manual cell execution)

---

## ✨ What This Framework Does

✅ Validates Phase 13 model is correctly structured  
✅ Checks 9D observation space  
✅ Verifies reward shaping formula  
✅ Tests end-to-end pipeline  
✅ Exports results  

---

**Created:** April 27, 2026  
**Status:** Ready to use  
**Next:** Run `python Simulation_4/scripts/phase13_test.py`
