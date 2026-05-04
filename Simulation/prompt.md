# Revised RL Simulation Prompt (Implemented)

This prompt reflects the implementation now present in the `Simulation` folder.

## Scope and constraints
- Keep Twitter and OpenAssistant as separate source domains.
- Use shared schema, never flatten both datasets into one anonymous pool.
- Keep confidence scores on all inferred labels.
- Restrict changes to `Simulation` only.

## Canonical action space
7-action environment action space:
1. `Ask_for_Information`
2. `Provide_Solution`
3. `Affective_Repair`
4. `Escalate_to_Human`
5. `Close_with_Feedback`
6. `Proactive_Update`
7. `Set_Expectation`

8-class labeling space (adds fallback):
8. `Unknown`

## Canonical turn-level schema
Each turn uses this schema (with nullable fields by role):
- `conv_id`
- `turn_id`
- `domain_source` (`twitter_cs` or `open_assistant`)
- `speaker_role` (`agent` or `customer`)
- `text`
- `text_length`
- `turn_index`
- `conv_length`
- `action_label`
- `action_probs`
- `action_confidence`
- `sentiment_score`
- `sentiment_confidence`
- `frustration_score`
- `frustration_confidence`
- `annotator_confidence`

## Implemented module map
Primary modular implementation is under:
- `Simulation/src/simulation_core/`

Data and labeling:
- `Simulation/src/simulation_core/data/data_pipeline_v2.py`
- `Simulation/src/simulation_core/labeling/llm_labeler.py`

Persona model:
- `Simulation/src/simulation_core/persona/persona_model.py`

Environment:
- `Simulation/src/simulation_core/env/pomdp_environment.py`

Reward model:
- `Simulation/src/simulation_core/rewards/reward_model.py`

Training pipeline (BC -> CQL-lite -> PPO-style fine-tune):
- `Simulation/src/simulation_core/training/training_pipeline.py`

Validation and bandits:
- `Simulation/src/simulation_core/evaluation/validation_suite.py`
- `Simulation/src/simulation_core/evaluation/bandits.py`

UI tools:
- `Simulation/ui/gold_annotation_tool.html`
- `Simulation/ui/simulator_dashboard.html`

Orchestration entrypoint:
- `Simulation/orchestrate_revised_pipeline.py`

Compatibility wrappers at top-level `Simulation/`:
- `data_pipeline_v2.py`
- `persona_model.py`
- `pomdp_environment.py`
- `reward_model.py`
- `training_pipeline.py`
- `validation_suite.py`
- `bandits.py`
- `rl_environment.py` (legacy compatibility forwarding)

## Execution flow
1. Run data pipeline:
   - `python Simulation/orchestrate_revised_pipeline.py`
   - Internally executes `run_pipeline()` in data pipeline module.

2. Outputs generated:
   - `Simulation/artifacts/revised_pipeline/labeled_data/twitter_labeled.csv`
   - `Simulation/artifacts/revised_pipeline/labeled_data/openassistant_labeled.csv`
   - `Simulation/artifacts/revised_pipeline/labeled_data/turn_index.csv`
   - `Simulation/reports/labeling_report.md`

3. Persona training and validation:
   - Produces `persona_validation.json` under revised artifacts.

4. Reward training and validation:
   - Produces `reward_weights.npy` and `reward_validation.json`.

5. Training pipeline:
   - BC checkpoint
   - CQL checkpoint
   - PPO-style policy checkpoint
   - `training_summary.json`

6. Validation suite:
   - Distribution fidelity, replay proxy, bandit comparisons
   - `validation_results.json`

## What changed from legacy implementation
- Legacy hand-coded MDP dynamics were replaced by POMDP module design.
- LLM usage moved out of per-step simulation transitions.
- Confidence-aware labeling added with low-confidence handling.
- Reward modeling moved to learned model (MaxEnt IRL with preference fallback).
- BC -> CQL-lite -> PPO-style training sequence added.
- Gold annotation and explainability dashboard tools added.

## Notes
- Existing notebooks can still import `CustomerSupportEnv` from `rl_environment.py`, which now maps to the revised POMDP environment class.
- New experiments should import modules directly from `Simulation/src/simulation_core/`.
