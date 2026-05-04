# Revised Simulation Modules

This folder contains the modular implementation for the revised architecture:

- `simulation_core/data/data_pipeline_v2.py`: Domain-separated labeling pipeline.
- `simulation_core/labeling/llm_labeler.py`: LLM + fallback labelers with confidence output.
- `simulation_core/persona/persona_model.py`: CVAE persona model and validation.
- `simulation_core/env/pomdp_environment.py`: POMDP environment with neural transition model.
- `simulation_core/rewards/reward_model.py`: MaxEnt IRL and preference fallback reward model.
- `simulation_core/training/training_pipeline.py`: BC -> CQL-lite -> PPO-style fine-tuning sequence.
- `simulation_core/evaluation/bandits.py`: Epsilon-greedy, UCB1, Thompson baselines.
- `simulation_core/evaluation/validation_suite.py`: Fidelity, replay proxy, and baseline comparisons.

UI tools:
- `../ui/gold_annotation_tool.html`
- `../ui/simulator_dashboard.html`

Entry point:
- `../orchestrate_revised_pipeline.py`
