# Strategy-Based Contextual Bandits: Short-Term vs Long-Term Optimization

This module implements **strategy-based contextual bandits** to compare with PPO and answer:

> **Can we achieve good performance with a fixed strategy selected at conversation start (short-term optimization) vs. adaptive multi-turn decisions (long-term optimization)?**

## The Problem Reframing

Traditional per-turn CB approach was failing because it treated each turn independently, ignoring multi-turn state transitions.

**Solution: Reframe as true single-step CB**

```
Traditional (Failed):
  Turn 0 → Decide AskInfo or Escalate
  Turn 1 → Decide ProvideSolution or Escalate  
  Turn 2 → Decide Close or Escalate
  ✗ No temporal reasoning, 0% resolution

Strategy-Based (Our Approach):
  Session Start → Decide ONE strategy for entire conversation
  Session End → Get outcome (resolved or not)
  ✓ True single-step CB problem
```

## Three Strategies

1. **Escalate Fast** (Strategy 0)
   - Escalate immediately (turn 3+ due to masking)
   - Good if escalation leads to resolution
   - Fast but risky

2. **Solve Patiently** (Strategy 1)
   - Ask info (turn 0-1)
   - Provide solutions (turn 2-3)
   - Never escalate
   - Longer but attempts self-service

3. **Hybrid Adaptive** (Strategy 2)
   - Use PPO policy (oracle baseline)
   - Optimal adaptive decisions
   - Computationally expensive

## Key Insight

**Short-term CB optimization** = "Pick one strategy that works well on average"
**Long-term MDP optimization** = "Adapt strategy to each customer based on their responses"

The comparison shows which is better for your problem.

## Usage

### 1. Train Strategy LinUCB

```bash
python -m Simulation_4.contextual_bandits.train_strategy_linucb \
  --collect-episodes 500 \
  --eval-episodes 200 \
  --alpha 1.0
```

**What it does:**
- Collects 500 episodes with random strategy selection
- Trains LinUCB to predict best strategy from initial observation + context
- Evaluates on 200 episodes

**Output:**
- `Simulation_4/artifacts/strategy_linucb_models/policy.pkl` - Trained model
- `Simulation_4/artifacts/strategy_linucb_models/results.json` - Evaluation metrics

### 2. Compare All Three Approaches

```bash
python -m Simulation_4.contextual_bandits.compare_all \
  --episodes 150 \
  --output Simulation_4/artifacts/comparison_results.json
```

**Compares:**
1. **PPO** - Adaptive multi-turn (long-term optimization)
2. **Strategy LinUCB** - Fixed strategy selection (short-term optimization)
3. **Per-Turn LinUCB** - Independent per-turn decisions (baseline)

**Output:** JSON with resolution rates, escalation rates, outcome distribution for each

## Expected Results

Based on customer support dynamics:

| Approach | Resolution | Reasoning |
|----------|-----------|-----------|
| PPO | ~48-69% | Adapts to customer responses, learns when to escalate |
| Strategy CB | ~30-50% | Good strategies but rigid, can't adapt |
| Per-Turn CB | ~0-10% | No temporal understanding at all |

**Why Strategy CB is competitive:**
- Sometimes a fixed strategy IS optimal (e.g., "always solve patiently")
- Less overfitting than PPO (fewer parameters)
- Faster inference

**Why PPO wins:**
- Can escape bad strategies (e.g., if customer clearly frustrated)
- Learns multi-step dependencies
- Longer episodes accepted if needed

## Code Example

```python
from Simulation_4.contextual_bandits.strategy_linucb import StrategyLinUCB, StrategyExecutor
from Simulation_4.env.support_env import SupportEnv

# Create environment
env = SupportEnv(...)

# Create policy
policy = StrategyLinUCB(n_strategies=3, d=9, alpha=1.0)

# Training loop
for episode in episodes:
    context = episode["initial_obs"]  # 9D observation
    strategy = np.random.randint(0, 3)  # Random exploration
    reward = episode["final_reward"]  # 1 if resolved, 0 otherwise
    
    policy.update(context, strategy, reward)

# Evaluation
obs, _ = env.reset()
strategy = policy.predict(obs)  # CB selects best strategy
executor = StrategyExecutor(strategy)

# Run entire episode with fixed strategy
for turn in range(max_turns):
    action = executor.get_action(obs, turn)
    obs, reward, done, _, info = env.step(action)
    if done:
        break
```

## Research Questions Answered

### Q1: Are adaptive policies necessary for customer support?

Compare PPO (~69%) vs Strategy CB (~40-50%):
- **If gap > 20%**: Yes, adaptation is critical → Use PPO
- **If gap < 10%**: No, fixed strategies work well → Can use CB for simplicity

### Q2: Which strategy is best?

Look at `strategy_distribution` in results:
- Dominant strategy = simple problem (just use that strategy)
- Mixed strategies = complex problem requiring adaptation

### Q3: Where does strategy matter most?

Stratified analysis by tier/persona:
- If Strategy CB wins on Free tier but PPO wins on Enterprise
- → Strategy depends on customer segment

## Files

- **`strategy_linucb.py`** - Core implementation
  - `StrategyContext`: Context representation
  - `StrategyExecutor`: Execute strategy throughout conversation
  - `StrategyLinUCB`: LinUCB for strategy selection

- **`train_strategy_linucb.py`** - Training script
  - Collect episodes with random strategies
  - Train LinUCB model
  - Evaluate policy

- **`compare_all.py`** - Comparison framework
  - Evaluate PPO, Strategy LinUCB, Per-Turn LinUCB
  - Generate detailed comparison report
  - Identify key differences

## Parameters

### `StrategyLinUCB` Parameters

- **`n_strategies`**: Number of strategies (default: 3)
- **`d`**: Feature dimension (default: 9, must match observation size)
- **`alpha`**: Exploration parameter (default: 1.0)
  - Higher = more exploration (try new strategies)
  - Lower = more exploitation (stick with known good strategies)

### Training Parameters

- **`--collect-episodes`**: Episodes to collect (default: 500)
  - More episodes = better coverage of context space
  - But more collection time

- **`--alpha`**: Exploration parameter
  - Try values: 0.5, 1.0, 2.0

## Advantages of This Approach

✅ **True Contextual Bandit**: Properly framed as single-step CB
✅ **Fair PPO Comparison**: Both see full problem scope
✅ **Practical**: Fixed strategies are interpretable and deployable
✅ **Research Contribution**: "Short-term vs long-term" framing is novel

## Future Work

1. **Strategy Library Expansion**
   - Add more sophisticated strategies
   - Learn strategy space with meta-learning

2. **Hybrid Approaches**
   - Select strategy + allow limited adaptation
   - Warm-start with CB, fine-tune with PPO

3. **Real-Time Strategy Switching**
   - Detect when current strategy fails
   - Switch to backup strategy mid-conversation

4. **Strategy Personalization**
   - Customer segment → best strategy
   - Per-persona models

## References

- Sajeev et al. (2021): Microsoft's CB approach to support bots
- Li et al. (2010): LinUCB algorithm paper
- Your contribution: "Short-term vs Long-term Optimization in Sequential Decision Problems"
