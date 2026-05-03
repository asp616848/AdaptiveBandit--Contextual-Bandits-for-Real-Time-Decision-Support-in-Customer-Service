# Contextual Bandits Experiment Suite

This folder contains the end-to-end experiment stack for fair Contextual Bandit vs PPO-style comparisons in the support simulator.

## What is implemented

- Dense per-turn reward shaping in `Simulation_4/env/reward_engine.py` and per-turn component logging in `Simulation_4/env/support_env.py`
- Action-level contextual bandits: LinUCB, Linear Thompson Sampling, Linear Epsilon-Greedy
- Strategy-level contextual bandits with 5 fixed strategies:
  - info-first
  - solve-first
  - emotion-first
  - escalate-early
  - balanced
- Horizon-controlled experiments: `T in {1,2,3,5,8,20}` (configurable)
- Truncated-hybrid baseline: CB controls first K turns, then balanced fallback strategy
- Metrics output for bandit analysis:
  - avg_per_turn_reward
  - immediate_regret (placeholder unless oracle is added)
  - early_turn_reward
  - sentiment_improvement
  - resolution_rate
  - mean_episode_reward

## Main runner

Run full suite:

```bash
python -m contextual_bandits.main \
  --collect-episodes 300 \
  --eval-episodes 200 \
  --horizons 1,2,3,5,8,20 \
  --seeds 1,2,3 \
  --algorithms linucb,thompson,epsilon
```

Quick sanity run:

```bash
python -m contextual_bandits.main \
  --collect-episodes 50 \
  --eval-episodes 50 \
  --horizons 1,3 \
  --seeds 1 \
  --algorithms linucb
```

## Visualization

```bash
python -m contextual_bandits.visualize --run-dir output/Contextual_bandit/<run_id>
```

Plots are saved under `output/Contextual_bandit/<run_id>/plots`.

## Output structure

- `output/Contextual_bandit/<run_id>/summary.csv`
- `output/Contextual_bandit/<run_id>/aggregate.csv`
- `output/Contextual_bandit/<run_id>/results.json`
- `output/Contextual_bandit/<run_id>/logs/*.jsonl`
- `output/Contextual_bandit/<run_id>/models/*`
- `output/Contextual_bandit/<run_id>/plots/*`

