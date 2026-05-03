from __future__ import annotations

from dataclasses import dataclass

from Simulation_4.env.support_env import SupportEnv
from Simulation_4.training.action_masking import ActionMaskedEnv
from Simulation_4.training.nlp_observation import NLPObservationWrapper
from Simulation_4.training.text_observation import TextOnlyObservationWrapper


@dataclass(frozen=True)
class BanditEnvConfig:
    artifacts_root: str = "Simulation_4/artifacts"
    use_nlp: bool = True
    use_nlg: bool = False
    use_masking: bool = True
    horizon: int | None = None
    text_only_observation: bool = False
    nlp_intent_model: str | None = None


def build_bandit_env(
    artifacts_root: str = "Simulation_4/artifacts",
    use_nlp: bool = True,
    use_nlg: bool = False,
    use_masking: bool = True,
    horizon: int | None = None,
    text_only_observation: bool = False,
    nlp_intent_model: str | None = None,
):
    """Build a SupportEnv configured for fair PPO / contextual bandit comparisons."""
    env = SupportEnv(artifacts_root=artifacts_root, nlg_enabled=bool(use_nlg))

    if horizon is not None:
        env.T_max = int(horizon)

    if use_nlp:
        if text_only_observation:
            env = TextOnlyObservationWrapper(env)
        else:
            env = NLPObservationWrapper(env, intent_model=nlp_intent_model)

    if use_masking:
        env = ActionMaskedEnv(env)

    return env


def build_contextual_bandit_env(
    artifacts_root: str = "Simulation_4/artifacts",
    use_nlp: bool = True,
    use_nlg: bool = False,
    use_masking: bool = True,
    horizon: int | None = None,
    text_only_observation: bool = False,
    nlp_intent_model: str | None = None,
    # contextual-bandit specific overrides (non-destructive)
    resolution_boost_logit: float = 0.15,
    max_required_slots: int = 2,
    dense_weight_overrides: dict | None = None,
    milestone_threshold: float = 0.7,
    milestone_bonus: float = 2.0,
):
    """Build a SupportEnv for contextual-bandit experiments and apply safe overrides.

    This function does NOT modify the SupportEnv source — it mutates the instance
    returned so experiments can be run with easier success dynamics and different
    dense reward weights without affecting other workflows.
    """
    env = SupportEnv(artifacts_root=artifacts_root, nlg_enabled=bool(use_nlg))

    if horizon is not None:
        env.T_max = int(horizon)

    # Apply subflow mean adjustments so fewer slots are required to reveal
    try:
        for k in list(env.subflow_mean_values.keys()):
            env.subflow_mean_values[k] = float(max_required_slots)
        # also propagate to state_engine view
        env.state_engine.subflow_mean_values = dict(env.subflow_mean_values)
    except Exception:
        pass

    # Bump baseline success logit so resolution probability increases
    try:
        env.state_engine.theta_0 = float(env.state_engine.theta_0) + float(resolution_boost_logit)
    except Exception:
        pass

    # Apply dense reward weight overrides if supplied
    if dense_weight_overrides:
        try:
            env.reward_engine.dense_weights.update({k: float(v) for k, v in dense_weight_overrides.items()})
        except Exception:
            pass

    # Soft success scaling based on information ratio instead of a hard zero gate.
    try:
        original_compute_p_success = env.state_engine.compute_p_success

        def scaled_compute_p_success(state: dict, action_name: str = "ProvideSolution") -> float:
            slot_tracker = getattr(env, "slot_tracker", None)
            if slot_tracker is None:
                return float(original_compute_p_success(state, action_name=action_name))

            revealed_count = len(getattr(slot_tracker, "revealed_slots", []))
            required_slots = max(int(max_required_slots), 1)
            info_ratio = float(min(max(revealed_count / required_slots, 0.0), 1.0))
            base_prob = float(original_compute_p_success(state, action_name=action_name))
            return float(base_prob * (0.3 + 0.7 * info_ratio))

        env.state_engine.compute_p_success = scaled_compute_p_success
    except Exception:
        pass

    # Middle-ground default dense rewards when no caller override is supplied.
    if not dense_weight_overrides:
        try:
            env.reward_engine.dense_weights.update(
                {
                    "delta_progress": 0.8,
                    "delta_info": 0.5,
                    "delta_sentiment": 0.8,
                    "delta_frustration": -0.3,
                }
            )
        except Exception:
            pass

    # Wrap the reward_engine.compute_per_turn_components to add a milestone bonus
    try:
        orig_compute = env.reward_engine.compute_per_turn_components

        def wrapped_compute(pre_state: dict, post_state: dict) -> dict:
            comps = orig_compute(pre_state, post_state)
            comps = dict(comps)
            comps["per_turn_reward"] = float(comps.get("per_turn_reward", 0.0)) + 0.05
            try:
                p_succ = env.state_engine.compute_p_success(post_state)
                if float(p_succ) > float(milestone_threshold):
                    comps["per_turn_reward"] = float(comps.get("per_turn_reward", 0.0)) + float(milestone_bonus)
                slot_tracker = getattr(env, "slot_tracker", None)
                revealed_count = len(getattr(slot_tracker, "revealed_slots", [])) if slot_tracker is not None else 0
                required_slots = max(int(max_required_slots), 1)
                info_ratio = float(min(max(revealed_count / required_slots, 0.0), 1.0))
                if info_ratio > 0.7:
                    comps["per_turn_reward"] = float(comps.get("per_turn_reward", 0.0)) + 1.5
            except Exception:
                pass
            return comps

        env.reward_engine.compute_per_turn_components = wrapped_compute
    except Exception:
        pass

    if use_nlp:
        if text_only_observation:
            env = TextOnlyObservationWrapper(env)
        else:
            env = NLPObservationWrapper(env, intent_model=nlp_intent_model)

    if use_masking:
        env = ActionMaskedEnv(env)

    return env