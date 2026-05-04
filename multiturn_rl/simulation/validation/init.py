from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

from simulation.env.support_env import SupportEnv


PolicyFn = Callable[..., int]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def artifacts_root() -> Path:
    return project_root() / "simulation" / "artifacts"


def phase9_artifacts_root() -> Path:
    out = artifacts_root() / "phase9"
    out.mkdir(parents=True, exist_ok=True)
    return out


def make_env(nlg_enabled: bool = False, subflow_filter: list[str] | None = None) -> SupportEnv:
    return SupportEnv(
        artifacts_root=str(artifacts_root()),
        nlg_enabled=bool(nlg_enabled),
        subflow_filter=subflow_filter,
    )


def obs_to_dict(obs: np.ndarray) -> dict[str, float]:
    obs = np.asarray(obs, dtype=np.float32)
    return {
        "subflow_id_norm": float(obs[0]),
        "tier_id_norm": float(obs[1]),
        "difficulty": float(obs[2]),
        "information": float(obs[3]),
        "progress": float(obs[4]),
        "frustration": float(obs[5]),
        "failed_streak_norm": float(obs[6]),
        "turn_count_norm": float(obs[7]),
        "resolved": float(obs[8]),
    }


def run_episode(
    env: SupportEnv,
    policy_fn: PolicyFn,
    seed: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    obs, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    transitions: list[dict[str, Any]] = []

    while not done:
        pre_state = dict(env.state)
        action = _call_policy(policy_fn, obs_to_dict(obs), pre_state, rng)
        pre_p_success = env.state_engine.compute_p_success(pre_state)
        pre_p_dropout = env.state_engine.compute_p_dropout(pre_state)

        obs, reward, done, _, info = env.step(int(action))
        total_reward += float(reward)

        post_state = dict(env.state)
        outcome = dict(info.get("last_transition_outcome", {}) or {})
        transitions.append(
            {
                "action": int(action),
                "action_name": env.ACTION_NAMES[int(action)],
                "reward": float(reward),
                "pre": pre_state,
                "post": post_state,
                "pre_p_success": float(pre_p_success),
                "pre_p_dropout": float(pre_p_dropout),
                "outcome": outcome,
            }
        )

    terminal_type = _terminal_type_from_state(env.state)
    return {
        "seed": int(seed),
        "total_reward": float(total_reward),
        "turn_count": int(env.state.get("turn_count", 0)),
        "state": dict(env.state),
        "terminal_type": terminal_type,
        "transitions": transitions,
    }


def _terminal_type_from_state(state: dict[str, Any]) -> str:
    if bool(state.get("resolved", 0)):
        return "resolved"
    if bool(state.get("escalated", 0)):
        return "escalated"
    if bool(state.get("dropped_off", 0)):
        return "dropout"
    return "timeout"


def _call_policy(policy_fn: PolicyFn, obs: dict[str, float], state: dict[str, Any], rng: np.random.Generator) -> int:
    try:
        return int(policy_fn(obs, state, rng))
    except TypeError:
        return int(policy_fn(obs, state))


def safe_check(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        out = fn()
        out.setdefault("check", name)
        out.setdefault("passed", False)
        return out
    except Exception as exc:
        return {
            "check": name,
            "passed": False,
            "value": f"{type(exc).__name__}: {exc}",
            "threshold": "No exception",
            "details": traceback.format_exc(limit=2),
        }
