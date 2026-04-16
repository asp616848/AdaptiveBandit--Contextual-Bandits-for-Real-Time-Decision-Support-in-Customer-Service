# Task Breakdown (Small Steps)

## Milestone A - Data Foundation

1. Implement `extract_minimal_abcd.py` parser.
2. Run parser on ABCD JSON.
3. Inspect generated `tasks_minimal.jsonl`.
4. Inspect generated `subflow_catalog.json`.
5. Implement robust success-label logic (beyond only `end_conversation`).
6. Generate success-label audit report (`strict`, `heuristic`, `ambiguous`).
7. Freeze top-K actions for first env version.
8. Map rare actions to `OTHER_ACTION` and record coverage stats.

## Milestone B - Environment Core

1. Implement deterministic environment class.
2. Add `reset` and `step` unit checks.
3. Add random rollout script or test.
4. Validate terminal conditions and rewards.

## Milestone C - PPO Training

1. Implement PPO training script.
2. Train for short run (sanity).
3. Train for full run.
4. Save best and final checkpoints.

## Milestone D - Evaluation and Baselines

1. Implement random baseline runner.
2. Implement heuristic baseline runner.
3. Evaluate PPO checkpoint.
4. Save report to `data/reports/`.

## Milestone E - Iterative Realism

1. Add stochastic transitions (one feature only) and benchmark.
2. Add information gate (one feature only) and benchmark.
3. Add frustration scalar (one feature only) and benchmark.
4. Re-run tests after each increment.
5. Re-train short run after each increment.
6. Compare with previous benchmark after each increment.

## Working Agreement

- Keep commits small and single-purpose.
- No hidden constants in code: move to config JSON.
- No realism feature unless prior milestone passes.
