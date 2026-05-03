from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure parent directory is in path for multiturn_rl imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multiturn_rl.simulation.env.support_env import SupportEnv
from multiturn_rl.simulation.training.action_masking import ActionMaskedEnv
from multiturn_rl.simulation.training.nlp_observation import NLPObservationWrapper

TextOnlyObservationWrapper = None


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
            if TextOnlyObservationWrapper is None:
                raise ImportError("text_only_observation=True requires multiturn_rl.simulation.training.text_observation")
            env = TextOnlyObservationWrapper(env)
        else:
            env = NLPObservationWrapper(env, intent_model=nlp_intent_model)

    if use_masking:
        env = ActionMaskedEnv(env)

    return env


def build_contextual_bandit_env_FINAL(
    artifacts_root: str = "Simulation_4/artifacts",
    use_nlp: bool = True,
    use_nlg: bool = False,
    use_masking: bool = True,
    horizon: int | None = None,
    text_only_observation: bool = False,
    nlp_intent_model: str | None = None,
    resolution_boost_logit: float = 0.25,
    max_required_slots: int = 2,
    dense_weight_overrides: dict | None = None,
    milestone_threshold: float = 0.7,
    milestone_bonus: float = 2.0,
):
    """Build a SupportEnv for contextual-bandit experiments with FINAL best settings.

    This builder uses the tried-and-tested parameter values that produced
    ~10% resolution on action CB and ~18% on strategy CB in early smoke tests.
    Does NOT modify the SupportEnv source — it mutates the instance returned.
    """
    env = SupportEnv(artifacts_root=artifacts_root, nlg_enabled=bool(use_nlg))

    if horizon is not None:
        env.T_max = int(horizon)

    # Reduce required information burden for this contextual-bandit setup.
    try:
        for key in list(env.subflow_mean_values.keys()):
            env.subflow_mean_values[key] = float(max_required_slots)
        env.state_engine.subflow_mean_values = dict(env.subflow_mean_values)
    except Exception:
        pass

    # Increase baseline success probability to improve signal frequency.
    try:
        env.state_engine.theta_0 = float(env.state_engine.theta_0) + float(resolution_boost_logit)
    except Exception:
        pass

    # Apply custom dense reward weights when provided by the caller.
    if dense_weight_overrides:
        try:
            env.reward_engine.dense_weights.update({k: float(v) for k, v in dense_weight_overrides.items()})
        except Exception:
            pass
    else:
        # Apply FINAL best-performing defaults when no override is supplied
        try:
            env.reward_engine.dense_weights.update(
                {
                    "delta_progress": 1.2,      # FINAL best: stronger progress signal
                    "delta_info": 0.6,          # FINAL best: moderate info signal
                    "delta_sentiment": 0.8,     # FINAL best: good sentiment weight
                    "delta_frustration": -0.3,  # FINAL best: moderate frustration penalty
                }
            )
        except Exception:
            pass

    # Add optional milestone reward for high success probability states.
    try:
        original_compute = env.reward_engine.compute_per_turn_components

        def wrapped_compute(pre_state: dict, post_state: dict) -> dict:
            components = original_compute(pre_state, post_state)
            try:
                p_success = env.state_engine.compute_p_success(post_state)
                if float(p_success) > float(milestone_threshold):
                    components = dict(components)
                    components["per_turn_reward"] = float(components.get("per_turn_reward", 0.0)) + float(milestone_bonus)
            except Exception:
                pass
            return components

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
