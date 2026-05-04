from __future__ import annotations

import numpy as np


def policy_always_escalate(obs, state) -> int:
    _ = obs, state
    return 3


def policy_random(obs, state, rng) -> int:
    _ = obs, state
    return int(rng.integers(0, 5))


def policy_threshold_escalate(obs, state, rng=None) -> int:
    _ = obs
    if float(state["frustration"]) > 0.7:
        return 3
    if rng is not None:
        return int(rng.choice(np.array([0, 1, 2, 4], dtype=np.int64)))
    return int(np.random.choice([0, 1, 2, 4]))


def policy_always_solve(obs, state) -> int:
    _ = obs, state
    return 1


def policy_document_guided(obs, state) -> int:
    _ = obs
    frustration = float(state["frustration"])
    information = float(state["information"])
    progress = float(state["progress"])
    failed_streak = int(state["failed_streak"])

    if failed_streak >= 4:
        return 3
    if progress >= 0.80:
        return 4
    if frustration > 0.50 and failed_streak > 0:
        return 2
    if information < 0.60:
        return 0
    return 1


POLICY_REGISTRY = {
    "always_escalate": policy_always_escalate,
    "random": policy_random,
    "threshold_escalate": policy_threshold_escalate,
    "always_solve": policy_always_solve,
    "document_guided": policy_document_guided,
}
