# Strategy vs Action: Understanding the Contextual Bandit Approaches

## Quick Answer

**Action** = One of 5 discrete choices in a single turn
**Strategy** = A decision rule for an entire conversation

```
Actions (per-turn):     [0: AskInfo, 1: ProvideSolution, 2: AffectiveRepair, 3: Escalate, 4: Close]
Strategies (full conversation): [0: EscalateFast, 1: SolvePatient, 2: HybridAdaptive]
```

## The Problem We Found

Your earlier per-turn LinUCB got **0% resolution** because it had a critical bug:

❌ **OLD Per-Turn LinUCB (BROKEN)**
```python
# train_linucb.py line 60-65
action = env.action_space.sample()  # Sample completely random action 0-5
obs, reward, done, _, info = env.step(action)

# Problem: Escalate (3) is masked before turn 3!
# Turn 0: Random samples might try action=3 (Escalate)
#         ↓ Gets silently converted to action=0 (AskInfo)
#         ↓ Training signal: "(obs, action=3, reward)" but env executed action=0!
#         ↓ CORRUPTED DATA → 0% resolution
```

**What happens:**
1. Training tries `action=3` at turn 0 (invalid)
2. Env silently overrides to `action=0`
3. Training records: `(obs, action=3, reward_for_action_0)`
4. Model learns: "Taking Escalate at turn 0 gives good reward" ❌ FALSE

---

## Three Approaches (Correct Comparison)

### 1️⃣ **PPO - Adaptive Multi-Turn (Baseline)**

```
Start → Obs → PPO → Action (respects mask) → Reward → State transition
              ↓     (learns what action works for THIS state)
         Repeat 1-20 turns
```

**How it works:**
- Sees observation, predicts best action for THIS specific state
- Takes action 0-4 (respecting mask)
- Gets reward, observes state change
- Learns dependencies between turns

**Expected:** ~48-69% resolution (adapts to each customer)

---

### 2️⃣ **Per-Turn LinUCB (Masked Actions) - Independent Per-Turn Decisions**

```
Start → Turn 0: CB picks action (respects mask) → Reward
        Turn 1: CB picks action (respects mask) → Reward  
        Turn 2: CB picks action (respects mask) → Reward
               ...
```

**How it works:**
- At each turn, LinUCB selects best action from valid actions only
- Each turn is an INDEPENDENT decision (ignores turn history)
- Gets per-turn reward signal
- Learns "what works at turn N" separately for each N

**Expected:** ~10-30% resolution (no multi-turn reasoning)

**Why it's still limited:**
- Treats turn 0 independently from turn 1
- Doesn't understand: "If I AskInfo at turn 0, customer will be more satisfied at turn 1"
- Decision at turn N can't depend on what customer said at turn N-1

---

### 3️⃣ **Strategy LinUCB - Fixed Strategy Selection**

```
Upfront Decision:
┌─ Strategy 0: Escalate fast
│  └ Turn 0: AskInfo → Turn 1: AskInfo → Turn 2: AskInfo → Turn 3+: Escalate
│
├─ Strategy 1: Solve patiently  
│  └ Turn 0: AskInfo → Turn 1: AskInfo → Turn 2: ProvideSolution → Turn 3: ProvideSolution → Turn 4+: Close
│
└─ Strategy 2: Hybrid adaptive
   └ Use PPO policy (oracle)

CB Decides → Which strategy? → Execute deterministically → Final outcome
```

**How it works:**
- LinUCB sees INITIAL observation
- Picks ONE strategy for entire conversation
- Strategy deterministically selects actions 0-4 at each turn
- One reward at end: Did it resolve?

**Expected:** ~30-50% resolution (semi-adaptive through strategy diversity)

---

## Key Insight: Why Does Masking Matter?

**Action Masking Rule:**
```
turn < 3:  Block action=3 (Escalate)  → Valid actions: [0,1,2,4]
turn >= 3: Allow all actions           → Valid actions: [0,1,2,3,4]
```

### Without Masking (BROKEN - Old Per-Turn LinUCB)
```python
# During training: Try random action (0-4)
action = env.action_space.sample()  # Might be 3 (Escalate) at turn 0!

# Environment: Silently override to action 0
if not action_mask[action]:
    action = valid_actions[0]  # action 0 is always valid

# Training records WRONG signal:
# "Took action=3, got this reward" but env actually took action=0
```

Result: **Model learns falsehoods → Useless policy**

### With Masking (CORRECT - New Masked Per-Turn LinUCB)
```python
# During training: Sample ONLY valid actions
action_mask = env.action_masks()  # [T, T, T, F, T] at turn 0
action = np.random.choice(np.where(action_mask)[0])  # Only 0,1,2,4

# Environment: Always valid, no override needed
action = whatever we sampled

# Training records CORRECT signal:
# "Took action=0, got this reward"
```

Result: **Model learns truth → Useful policy**

---

## Comparison Table

| Aspect | PPO | Masked Per-Turn LinUCB | Strategy LinUCB |
|--------|-----|----------------------|-----------------|
| **Decision** | Per-turn, adaptive | Per-turn, adaptive | Per-conversation, fixed |
| **Action Space** | 5 actions (masked) | 5 actions (masked) | 3 strategies |
| **Uses State Transitions** | ✓ YES | ✗ NO | ✓ IMPLIED |
| **Can Adapt Mid-Conversation** | ✓ YES | ✗ NO (independent) | ✗ NO (locked at start) |
| **Expected Resolution** | ~48-69% | ~10-30% | ~30-50% |
| **Interpretability** | Medium | Low | HIGH ✓ |
| **Training Data Quality** | Good | Good (masked) | Good |

---

## The Strategy Execution (All 5 Actions Used!)

Your StrategyExecutor is correctly using all 5 actions:

```python
class StrategyExecutor:
    def get_action(self, obs, turn):
        if self.strategy == 0:  # Escalate fast
            if turn >= 3:
                return 3  # Escalate ← Uses action 3
            else:
                return 0  # AskInfo ← Uses action 0
        
        elif self.strategy == 1:  # Solve patiently
            if turn < 2:
                return 0  # AskInfo ← Uses action 0
            elif turn < 4:
                return 1  # ProvideSolution ← Uses action 1
            else:
                return 4  # Close ← Uses action 4
                # Never uses action 3 (Escalate)
        
        elif self.strategy == 2:  # Hybrid adaptive
            action, _ = ppo_model.predict(obs)
            return action  # Can be any 0-4 (respects masking inside PPO)
```

✅ Strategy 0 uses: [0, 3]  
✅ Strategy 1 uses: [0, 1, 4]  
✅ Strategy 2 uses: [0, 1, 2, 3, 4] (via PPO)  
✅ All strategies respect action masking (e.g., Strategy 0 doesn't escalate until turn 3)

---

## Why Strategy LinUCB > Per-Turn LinUCB (Even Masked)

Even with action masking fixed, per-turn LinUCB fails because:

```
Turn 0: Predict best action for fresh customer
        → LinUCB decides: AskInfo

        Customer: "I want refund, my account is locked"
        ↓ State updates, confidence ↑, frustration ↑
        
Turn 1: Predict best action for UPDATED customer
        → LinUCB decides: ProvideSolution (based on turn 0 training)
        
        But customer is frustrated! Should de-escalate first
        ↓ Wrong decision
        
        PROBLEM: Per-turn LinUCB trained on generic "turn 1 contexts"
                 Never trained on "frustrated customer at turn 1"
                 Because training was from random exploration!
```

**Strategy LinUCB avoids this:**
```
Strategy 1: "Always ask early, solve later, never escalate"

Turn 0: AskInfo
Turn 1: AskInfo (even if frustrated)
Turn 2: ProvideSolution (now we try solving)
Turn 3: ProvideSolution
Turn 4+: Close

Benefit: Follows coherent plan, adapts indirectly through strategy diversity
Cost: Can't adapt mid-conversation
```

---

## Testing Plan

**Run in order:**

```bash
# 1. Strategy LinUCB (you created this)
python -m Simulation_4.contextual_bandits.train_strategy_linucb \
  --collect-episodes 500 --eval-episodes 200

# 2. Masked Per-Turn LinUCB (NEW - corrected version)
python -m Simulation_4.contextual_bandits.train_masked_linucb \
  --collect-episodes 500 --eval-episodes 200

# 3. Compare all three
python -m Simulation_4.contextual_bandits.compare_all \
  --episodes 200 \
  --per-turn-linucb-model "Simulation_4/artifacts/masked_linucb_models/models"
```

**Expected Results:**
```
PPO:                    ~48-69%  (adaptive, learns multi-turn dependencies)
Strategy LinUCB:        ~30-50%  (fixed but diverse strategies)
Masked Per-Turn LinUCB: ~10-30%  (independent turns, even if masked)
Old Per-Turn LinUCB:    ~0%      (corrupted training data)
```

---

## Summary

✅ **Strategies ARE using all 5 actions correctly**  
✅ **Actions are properly masked before/after turn 3**  
❌ **Old Per-Turn LinUCB was BROKEN** (no masking during training)  
✅ **New Masked Per-Turn LinUCB fixes this**  
✅ **Strategy LinUCB is fundamentally different** (full-conversation decision, not per-turn)

The difference is architectural, not just in action selection:
- **Per-turn CB** = "Best independent decision at each turn"  
- **Strategy CB** = "Best consistent approach for whole conversation"  
- **PPO** = "Best adaptive decision considering full history"
