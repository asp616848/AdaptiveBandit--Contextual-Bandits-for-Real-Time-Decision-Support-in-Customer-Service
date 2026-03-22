# Research-Grade Support Simulator Context

Last updated: 2026-03-21
Owner: RL/AML project
Scope: Hybrid hidden-state simulator for escalation-policy RL (SaaS support)

## Problem Framing
- Objective: Learn policy minimizing expected economic cost-to-serve.
- Action set (core): AskInfo, ProvideSolution, AffectiveRepair, Escalate, Close.
- Extended action set (optional): 7-action taxonomy from existing pipeline.
- Environment type: Sequential decision process with multi-turn uncertainty.

## Non-Negotiable Design Constraints
- Simulator must be state-based (not pure LLM/RAG).
- Explicit hidden state, transition dynamics, and reward equations are mandatory.
- LLM/RAG are optional realism layers only; they cannot update latent state directly.

## Data Usage Contract
- Twitter dataset role:
  - estimate unresolved/failure proxies
  - derive escalation-like signals and frustration proxies
  - estimate turn-level persistence/no-response patterns
- OpenAssistant (OASST1) role:
  - map response quality to success/resolution likelihood
  - calibrate relationship between quality and satisfaction

## Economic Objective
- Cost = Escalation Cost + Expected Churn Loss + Inference/Turn Cost
- Reward should be the signed negative of cost plus success bonus.

## Candidate Hidden State (Task 1)
- Static sampled-at-reset:
  - customer_tier in {Free, Pro, Business, Enterprise}
  - problem_type in controlled categorical domain
  - difficulty in [0,1]
  - persona parameters (patience, sensitivity, failure_tolerance)
- Dynamic per-turn:
  - frustration in [0,1]
  - information_collected in [0,1]
  - resolution_probability in [0,1]
  - turn_count in {0,1,2,...,T_max}
  - resolved in {0,1}

## Modeling Principles
- Keep state minimal and interpretable for stable RL.
- Use monotonic effects where possible:
  - info_collected should increase success chance.
  - higher difficulty should reduce success chance.
  - repeated failed attempts should increase frustration nonlinearly.
- Ensure bounded updates and numerically stable equations.

## Interfaces Planned
- Environment API:
  - reset(seed=None) -> observation
  - step(action) -> observation, reward, done, info
- Observation to agent should be structured numeric state (no raw text required).

## Validation Targets (to implement later)
- Turn-count distribution realism
- Resolution and escalation rate realism
- Frustration trajectory plausibility
- Reward/cost decomposition sanity checks

## Open Decisions To Resolve In Task 2+
- Exact transition equations per action
- Exact churn function form and calibration
- Tier-dependent escalation costs and churn losses
- Whether to expose full latent state or a partial observation to policy
