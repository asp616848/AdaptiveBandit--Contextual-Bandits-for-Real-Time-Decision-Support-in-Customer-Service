# Simulation_4 Approach 1 (Fresh Rebuild)

This folder is a clean, minimal, RL-first rebuild path.

Goal: build a trainable customer-support simulator from ABCD with a simple deterministic core before adding realism.

## Why this exists

The previous simulator became too complex too early (many latent variables and stochastic terms). This rebuild starts with a strong action-consequence loop so PPO can learn reliably.

## What is inside

- `docs/MASTER_PLAN_APPROACH1.md`: full phased plan and acceptance criteria.
- `docs/TASK_BREAKDOWN.md`: small executable tasks in order.
- `src/data_pipeline/extract_minimal_abcd.py`: creates minimal task dataset from ABCD.
- `src/env/simple_support_env.py`: deterministic minimal Gymnasium environment.
- `src/train/train_ppo.py`: PPO training script for the simple environment.
- `src/eval/evaluate_policy.py`: evaluation script and baseline comparison.
- `configs/reward_config.json`: reward and termination config.
- `configs/train_config.json`: PPO defaults.
- `tests/test_env_smoke.py`: environment sanity checks.

## Phase order (strict)

1. Data extraction and validation.
2. Deterministic environment + smoke tests.
3. PPO training + baseline comparisons.
4. Evaluation report.
5. Add exactly one realism feature at a time.

## Non-goals in initial version

- Persona modeling
- Churn economics
- Complex frustration equations
- RAG-conditioned transitions

These will be considered only after core RL convergence is proven.
