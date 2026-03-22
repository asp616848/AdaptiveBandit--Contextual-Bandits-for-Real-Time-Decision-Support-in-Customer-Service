# Simulation and Implementation Plan (Improved)

## Project Context
This plan defines a defensible simulation-to-training pipeline for customer support policy learning using:
- Twitter dataset: `B:\College\RL\AdaptiveBandit--Contextual-Bandits-for-Real-Time-Decision-Support-in-Customer-Service\twitter`
- OpenAssistant dataset: `B:\College\RL\AdaptiveBandit--Contextual-Bandits-for-Real-Time-Decision-Support-in-Customer-Service\OpenAssistant Conversations Dataset`

The objective is to train and evaluate routing policies (bot, human, hybrid, and finer action variants) with realistic user emotion and escalation dynamics.

## 1. Design Philosophy

### 1.1 Why RAG-Grounded Simulation
A simulator generated without grounding can produce unrealistic emotional trajectories. This plan uses retrieval-augmented generation (RAG) so each synthetic turn is conditioned on similar real turns and persona-compatible trajectories.

Benefits:
- Auditability: synthetic turn can be traced to retrieved real neighbors.
- Fidelity: sentiment/escalation patterns stay close to observed data.
- Defensibility: simulator quality can be statistically validated on held-out data.

### 1.2 Core Thesis
Contextual bandits should perform competitively on short interactions, while full MDP RL should outperform on longer conversations where credit assignment across turns matters.

## 2. Goals and Success Criteria

### 2.1 Core Goals
- Build a reproducible, RAG-grounded dialogue simulator.
- Infer and validate action labels from real agent utterances.
- Train bandit and RL policies on a shared state/action interface.
- Validate simulator fidelity and policy transfer with offline evaluation.

### 2.2 Success Criteria
- Full turn-level schema populated for all simulated episodes.
- Action-label agreement target: Cohen's kappa >= 0.70 on a human-annotated sample.
- Simulator fidelity passes pre-defined statistical thresholds.
- RL policy shows best gains on medium/long-horizon trajectories.

## 3. Data and Schema Strategy

## 3.1 Data Roles
- Twitter: real customer support dynamics, escalation language, thread behaviors.
- OpenAssistant: broader dialogue style coverage, richer text patterns, robustness for language features.

## 3.2 Unified Turn-Level Schema
Every turn record should include:
- Identifiers: `conversation_id`, `thread_id`, `turn_id`, `speaker`.
- Text: `utterance_text`, `retrieved_neighbours`.
- Core state: `sentiment_score`, `sentiment_delta`, `frustration_proxy`, `escalation_likelihood`, `resolution_probability`, `turn_count`.
- Extended state: `emotion_category`, `patience_score`, `peak_frustration_seen`, `consecutive_no_progress_turns`.
- Action fields (agent turns): `action_label`, `action_id`, `action_confidence`.
- Persona fields: `persona_cluster`, `persona_confidence`.
- Reward/outcome fields: `turn_reward`, `is_terminal`, `terminal_outcome`.

## 3.3 Extended Action Space
Canonical action set (recommended 7 actions):
1. `Ask_for_Information`
2. `Provide_Solution`
3. `Affective_Repair`
4. `Escalate_to_Human`
5. `Close_with_Feedback`
6. `Proactive_Update`
7. `Set_Expectation`

## 4. Action Labeling Protocol

### 4.1 Label Inference
Use an LLM classifier to assign exactly one action label to each agent utterance using short context windows (last 2 turns + current agent turn).

### 4.2 Quality Control
- Manually annotate a stratified sample (recommended 200 turns).
- Compare LLM vs human labels.
- Track confusion matrix and Cohen's kappa.
- Refine label prompt and retry until quality target is reached.

## 5. Persona Extraction and Clustering

### 5.1 Persona Features
Compute conversation-level user behavior features:
- average sentiment
- sentiment volatility
- escalation event rate
- average user utterance length
- first-turn frustration
- recovery rate after negative sentiment trough
- dropout flag
- question ratio

### 5.2 Clustering Method
Use GMM with standardized features and BIC-based selection for number of clusters. Keep soft assignments to model mixed user tendencies.

### 5.3 Operational Personas
Target interpretable personas:
- Cooperative
- Escalation-prone
- Impatient
- Silent drop-off

## 6. RAG-Based Simulation Architecture

### 6.1 Dual Index Design
- Turn index: nearest-neighbor retrieval of similar user turns by state/persona context.
- Conversation index: trajectory-level retrieval at episode start for scenario initialization.

### 6.2 Per-Turn Generation Loop
At each decision step:
1. Build query from current state, persona, and previous action.
2. Retrieve top-k real neighbors with persona-compatible filtering.
3. Generate next user turn constrained by state and retrieved examples.
4. Extract updated state features from generated text.
5. Compute reward and terminal status.

### 6.3 Safety Constraints
- If escalation likelihood exceeds threshold (for example > 0.85), force legal action set to escalation-safe actions.
- Apply hard filters to remove unsafe, off-domain, or malformed generated turns.

## 7. Transition and Reward Modeling

### 7.1 Transition Model
Train a model for `f(s_t, a_t, persona) -> s_{t+1}` on historical tuples.

Recommended checks:
- Hold-out evaluation on 20% tuples.
- MAE/RMSE per state dimension.
- With-persona vs without-persona ablation.

### 7.2 Shaped Reward
Use a shaped reward with dominant terminal outcomes:
- Positive for sentiment/frustration improvement and resolution.
- Negative for delay, extra turns, and escalation/abandonment.

Example:
`R_t = 0.3 * delta_sentiment + 0.2 * delta_frustration - 0.05 * turn_penalty + action_bonus + terminal_reward`

### 7.3 Sensitivity Sweeps
Run sweeps over:
- Discount factor `gamma` in {0.7, 0.8, 0.9, 0.95}
- Turn penalty in {0.02, 0.05, 0.10}
- Escalation penalty in {0.3, 0.5, 0.8}

Use >=5 seeds per setup and report mean +/- std.

## 8. Validation Framework (5 Dimensions)

### 8.1 Statistical Fidelity
- Sentiment distribution match (KS test)
- Escalation-rate match (chi-square)
- Conversation-length divergence (KL)
- Trajectory shape distance (DTW)
- Action distribution divergence (KL)

### 8.2 Transition Accuracy
- Per-dimension MAE/RMSE on held-out transitions.
- Error breakdown by action type and persona.

### 8.3 Persona Fidelity
- Verify expected differences in escalation rate, dropout rate, utterance length, and response to specific actions.

### 8.4 Business Validity
- Resolution rate
- Escalation cost proxy
- Time-to-resolution proxy
- CLV retention proxy
- ROI estimate under realistic volume assumptions

### 8.5 Ethics and Risk Controls
- Bias audit on persona assignments and action outcomes.
- Privacy safeguards (PII removal before indexing).
- Escalation safety constraints to avoid harmful under-escalation.

## 9. Policy Training Plan

### 9.1 Baselines and Main Models
- Random and rule-based baselines
- LinUCB and Thompson Sampling
- PPO as primary full-RL model
- Optional DQN comparison where useful

### 9.2 Warm-Start Strategy
Use a contextual bandit policy to warm-start RL policy weights for faster and more stable convergence.

### 9.3 Curriculum
1. Offline pretraining: behavior cloning and transition pretraining.
2. Simulator training: full RL with periodic checkpoint evaluation.
3. Optional online fine-tuning: conservative updates if live traffic exists.

### 9.4 Evaluation Protocol
- Checkpoints every fixed training steps.
- Report metrics by dialogue length buckets: short, medium, long.
- Compare RL vs bandit vs behavior cloning vs random.
- Add offline transfer checks on held-out real logs.

## 10. Implementation Timeline

### Week 1
- Data audit, cleaning, and schema mapping.
- Build turn-level records and initial text features.

### Week 2
- Action labeling pipeline and inter-rater validation.
- Persona feature extraction and GMM clustering.

### Week 3
- Transition model training and error analysis.
- Build dual-index RAG retrieval layer.

### Week 4
- Complete simulator loop (`reset`, `step`, reward, terminal logic).
- Run simulator fidelity validation suite.

### Week 5
- Train bandit baselines and establish benchmark metrics.

### Week 6
- Train PPO and run ablations (with/without persona, with/without uncertainty).

### Week 7
- Sensitivity analysis, business-impact calculations, and policy transfer tests.

### Week 8
- Final reporting package (tables, plots, methods, limitations, reproducibility checklist).

## 11. Deliverables
- Improved implementation plan (this document).
- Data audit and schema map.
- Action-label quality report (including kappa).
- Persona clustering report (including BIC and cluster interpretations).
- Validated simulator with test report across 5 dimensions.
- Policy benchmarking report (bandit vs RL) with sensitivity and ablations.
- Final business-impact summary and project-ready figures.

## 12. Immediate Next Steps
1. Create a dataset schema inventory for both source datasets.
2. Implement action-label classifier script and annotation template.
3. Build persona-feature table and run GMM/BIC selection.
4. Implement the turn-level simulator interface with RAG retrieval hooks.
5. Run first benchmark experiment: Rule vs LinUCB vs Thompson.
6. Start PPO warm-start training from best bandit policy.

---
Owner: RL project team
Status: Updated and ready for execution
Last updated: 2026-03-18
