# Phase 13 LUMO Evaluation - Quick Run Guide

## 📊 What This Does

Runs your **Phase 13 model** against **100+ LUMO-based realistic customer service scenarios** and measures everything:

### PRIMARY METRIC
- **Resolution Rate** (%) — % of conversations where agent successfully resolved the issue

### ALL METRICS COLLECTED
1. **Terminal Outcomes**
   - Success / Escalation / Dropout / Timeout
   
2. **Reward Distribution**
   - Mean, std, min, max, median cumulative rewards

3. **Efficiency**
   - Average turns to resolution
   - Turn range (min/max)

4. **Customer Satisfaction Proxy**
   - Based on: success vs escalation/dropout + turn efficiency
   - Score: 0.0-1.0

5. **By Tier Performance**
   - Free / Pro / Business / Enterprise
   - Resolution rate per tier
   - Escalation rate per tier

6. **By Persona Performance**
   - high_engagement_resolver
   - low_engagement_resolver
   - silent_dropout
   - escalation_prone

---

## 📈 VISUALIZATIONS

The script automatically generates 8 professional plots:

1. **terminal_outcomes_pie.png** — Pie chart of success/escalation/dropout/timeout split
2. **performance_by_tier.png** — Bar chart comparing tiers (resolution vs escalation)
3. **performance_by_persona.png** — Bar chart comparing personas
4. **reward_distribution.png** — Histogram + KDE of all rewards (mean/median lines)
5. **turns_to_resolution.png** — Distribution of turns taken to resolve cases
6. **episode_rewards_trajectory.png** — Time series showing reward per episode + rolling average
7. **satisfaction_distribution.png** — Customer satisfaction proxy distribution
8. **terminal_outcomes_by_tier_stacked.png** — Stacked bar showing outcomes per tier

---

## 📋 STATISTICAL ANALYSIS

Also generates **automatic analysis report** with:

- Overall performance vs random baseline (4x improvement expected)
- Reward statistics (mean, std, coefficient of variation)
- Efficiency metrics (turns to resolution, efficiency ratio)
- Satisfaction interpretation (poor/fair/good/very good)
- Tier-by-tier breakdown
- Persona-by-persona breakdown
- Key observations (model behavior flags)

---

## ⏱️ Runtime Estimates

| Episodes | Time | Quality |
|----------|------|---------|
| 25 | 2-5 min | Quick smoke test |
| 100 | 8-20 min | Full baseline ✓ RECOMMENDED |
| 250 | 20-60 min | Comprehensive |
| 500+ | 60+ min | Production-grade |

**⚠️ First run may be slower** (initializing LLM backend, etc.)

---

## 🚀 How to Run

### Quick Test (25 episodes) + Plots
```bash
cd /Users/tishabhavsar/RL_Project /AdaptiveBandit--Contextual-Bandits-for-Real-Time-Decision-Support-in-Customer-Service

python Simulation_4/scripts/phase13_lumo_evaluation.py --episodes 25
```

### Full Evaluation (100 episodes) - RECOMMENDED
```bash
python Simulation_4/scripts/phase13_lumo_evaluation.py --episodes 100
```

### Comprehensive (250+ episodes)
```bash
python Simulation_4/scripts/phase13_lumo_evaluation.py --episodes 250
```

### Custom Output Paths
```bash
python Simulation_4/scripts/phase13_lumo_evaluation.py --episodes 100 \
  --output my_results.json \
  --plots ./my_plots_folder
```

---

## 📋 Expected Output

**Console Output:**

```
██████████████████████████████████████████████████████████████████████
  PHASE 13 LUMO EVALUATION
██████████████████████████████████████████████████████████████████████

...running episodes...

██████████████████████████████████████████████████████████████████████
  EVALUATION SUMMARY
██████████████████████████████████████████████████████████████████████

PRIMARY METRIC - RESOLUTION RATE:
  🎯 Resolution: 78.5%

[all metrics printed]

✓ Results saved to: phase13_evaluation_100_20260427.json

======================================================================
STATISTICAL ANALYSIS
======================================================================

1. OVERALL PERFORMANCE
  Total Episodes: 100
  Primary Metric (Resolution): 78.50%
  Expected Resolution for Random Policy: ~25%
  Improvement Factor: 3.14x

2. REWARD ANALYSIS
  Mean Reward: +0.2340
  Std Dev: 0.1560
  Range: [-0.1250, +0.8920]

[full analysis with observations]

✓ Analysis saved to: phase13_evaluation_100_20260427_analysis.txt

📊 Generating visualizations...
  ✓ terminal_outcomes_pie.png
  ✓ performance_by_tier.png
  ✓ performance_by_persona.png
  ✓ reward_distribution.png
  ✓ turns_to_resolution.png
  ✓ episode_rewards_trajectory.png
  ✓ satisfaction_distribution.png
  ✓ terminal_outcomes_by_tier_stacked.png

✓ All visualizations saved to: .

📊 All outputs generated successfully!
```

---

## 📁 Output Files

For each run, you get:

1. **phase13_evaluation_100_TIMESTAMP.json**
   - Complete raw data (all 100 episode logs)
   - All aggregated metrics
   - Machine-readable for further analysis

2. **phase13_evaluation_100_TIMESTAMP_analysis.txt**
   - Human-readable statistical analysis
   - Key observations and flags
   - Compared to baselines
   - Can be shared directly

3. **8 PNG Plot Files**
   - Professional quality (150 DPI)
   - Ready for presentations/reports
   - Shows all key insights visually

**Quick view:**
```bash
# See the analysis
cat phase13_evaluation_100_*_analysis.txt

# View plots (Mac)
open *.png

# Extract metrics from JSON
jq '.resolution_rate' phase13_evaluation_100_*.json
```

---

## ✅ Next Steps

1. **Run quick test first** (25 episodes, ~5 min)
   ```bash
   python Simulation_4/scripts/phase13_lumo_evaluation.py --episodes 25
   ```

2. **If looks good**, run full eval (100 episodes, ~20 min)
   ```bash
   python Simulation_4/scripts/phase13_lumo_evaluation.py --episodes 100
   ```

3. **Review outputs:**
   - Look at console summary
   - Read the analysis file
   - View the plots

4. **Compare metrics against Phase 10** if available
   - Check resolution rate improvement
   - Check satisfaction proxy gain
   - Check escalation vs resolution balance

---

## 🔧 Configuration

**Model Path**: `RL Out/models/best_model.zip` (Phase 13 from best_model/)

**Environment**: SupportEnv with NLG enabled for realistic scenario generation

**Scenarios**: LUMO-based from `scenario_templates.json` 

**Deterministic**: Using deterministic policy predictions (no randomness in action selection)

**Analysis**: Includes comparisons to random baseline (~25% resolution expected)

---

## ⚠️ Troubleshooting

**Error: "No such file or directory"**
- Make sure you're in the repo root (with trailing space in folder name!)
- Check model exists: `ls -lh RL\ Out/models/best_model.zip`

**Plots not generating**
- Make sure matplotlib/seaborn installed: `pip install matplotlib seaborn`
- Check write permissions in output directory

**No LLM responses (slow performance)**
- Check if Ollama is running (if using local LLM)
- May fall back to template-based responses

**Memory issues**
- Start with small episode count (25)
- Increase incrementally

---

## 📈 Interpretation Guide

### Resolution Rate
- **60-90%**: Excellent, model is learning well
- **40-60%**: Good, but room for improvement
- **<40%**: Poor, model needs more training
- **Random baseline**: ~25%

### Turns to Resolution
- **2-3 turns**: Excellent (fast resolution)
- **3-5 turns**: Good
- **>5 turns**: Inefficient (customer getting frustrated)

### Satisfaction Proxy
- **>0.75**: Very good experience
- **0.50-0.75**: Acceptable
- **<0.50**: Poor experience

### Escalation Rate
- **<20%**: Good (agent can handle most)
- **20-40%**: Moderate (some complex cases)
- **>40%**: High (agent escalates too much)

### By-Tier Differences
- Free tier usually harder (less context, external systems)
- Enterprise should have higher resolution (better support tools)

### By-Persona Differences
- high_engagement_resolver: Should have highest resolution
- escalation_prone: Should have higher escalation (by design)
- silent_dropout: Should have lower escalation (just leaves)

---

## 💡 Tips

- **First run**: Use 25 episodes to check if everything works
- **Production eval**: Use 100+ episodes for statistical significance
- **Comparisons**: Save multiple results to compare model versions
- **Debugging**: Use episode logs (in JSON) to trace specific failures
- **Sharing**: Send the analysis.txt + PNG plots (not the raw JSON)


