# Master Plan - Approach 1 (Simple First)

## Objective

Build a minimal, data-grounded simulator that is easy for RL to learn, then grow complexity gradually.

Core loop:

1. Sample task (subflow + required action sequence).
2. Agent takes action.
3. Environment returns immediate reward and updated progress.
4. Episode ends on success/escalation/early close/timeout.

## Folder Contract

- `src/data_pipeline`: ABCD parsing and dataset creation.
- `src/env`: environment implementation only.
- `src/train`: training entry points.
- `src/eval`: evaluation and baselines.
- `configs`: JSON config files (rewards, training).
- `data`: generated data artifacts for this approach.
- `tests`: smoke tests and regression checks.

## Data Contract

Input: ABCD JSON (`abcd_v1.1.json`) available outside git.

Output A (`tasks_minimal.jsonl`): one record per conversation.

- `convo_id`
- `split`
- `subflow`
- `actions_required` (ordered list)
- `success` (0/1)

Output B (`subflow_catalog.json`): one record per subflow.

- `subflow`
- `canonical_actions`
- `support_count`
- `empirical_success_rate`

## Phase Plan

### Phase 0 - Setup and Guardrails

Deliverables:

- Branch `new` for all work.
- This plan and task list in repo.
- Minimal config files.

Acceptance:

- A new contributor can open this folder and understand build order in 5 minutes.

### Phase 1 - Distill ABCD to Minimal Task Data

Work:

- Parse train/dev/test splits.
- Extract subflow and action sequence.
- Derive robust success label (not only `end_conversation`).
  - Start with `end_conversation`.
  - Add fallback heuristics and uncertainty flags for ambiguous cases.
- Build subflow catalog from successful conversations.
- Add action-vocab pruning policy.
  - Build global action frequencies.
  - Keep top-K frequent actions for Phase 2.
  - Map rare actions to `OTHER_ACTION` for controlled action-space size.

Acceptance:

- Output files created.
- Summary stats printed:
  - number of subflows
  - number of usable conversations
  - min/mean/max sequence length
- Success-label quality report generated:
  - exact `end_conversation` count
  - heuristic-labeled count
  - ambiguous count
- Action-vocab report generated:
  - full vocabulary size
  - top-K coverage percentage
  - rare-action percentage

### Phase 2 - Deterministic Environment (Learnable Core)

State:

- normalized subflow id
- normalized progress index
- normalized max sequence length
- normalized turns remaining

Actions:

- `ASK_INFO`
- `ESCALATE`
- `CLOSE`
- `DO_<specific_action>` for each action in global action vocabulary

Transition rules:

- Correct `DO_<action>` at current step: `+1`, advance step.
- Wrong `DO_<action>`: `-1`.
- `ESCALATE`: terminal `-2`.
- `CLOSE` before completion: terminal `-3`.
- Completion: terminal bonus `+5`.
- Timeout (`max_steps + slack`): terminal `-5`.

Acceptance:

- `tests/test_env_smoke.py` passes.
- Random rollout returns all terminal types in expected ranges.

### Phase 3 - PPO Training Baseline

Work:

- Train PPO on deterministic environment.
- Save model and metadata.
- Save reward curve and success curve.

Acceptance:

- Trained policy beats random policy by at least 2x success rate.
- Results stable across 3 seeds.

### Phase 4 - Evaluation

Metrics:

- success rate
- avg episode reward
- avg steps
- early close rate
- escalation rate
- timeout rate

Acceptance:

- evaluation JSON and text summary generated.
- confusion table over selected subflows generated.

### Phase 5 - Controlled Realism (One Change Per Iteration)

Add only one feature each cycle:

1. Stochastic transitions on correct action (`p` from data).
2. Information gate before solution actions.
3. Frustration as simple scalar.

Not-yet-implemented items tracked in this phase:

- Stochastic transitions are currently deterministic in Phase 2 core and must be added here.
- Information gate is not active in current core and must be added here.
- Frustration variable is not active in current core and must be added here.

Rules:

- Never add two realism features in same commit.
- Every realism addition must preserve learnability benchmark.

Acceptance:

- After each realism increment, trained policy still > random by 2x success.

## Definition of Done (Approach 1)

- Reproducible data extraction.
- Deterministic env that converges under PPO.
- Evaluation report showing clear policy improvement.
- Clean handoff docs for next complexity stage.

## Risks and Controls

Risk: action-space explosion.

- Control: keep only top-K frequent actions initially.

Risk: ambiguous success label.

- Control: start with explicit `end_conversation`; add heuristic fallback and log uncertain cases.

Risk: action-space explosion from long-tail actions.

- Control: enforce top-K action pruning with `OTHER_ACTION` bucket and track coverage.

Risk: reward hacking.

- Control: monitor terminal-type distribution and add penalties only when observed exploit appears.
