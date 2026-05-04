from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import PPO

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Simulation_4.env import SupportEnv
from Simulation_4.training.action_masking import ActionMaskedEnv
from Simulation_4.validation.baseline_policies import (
    policy_always_escalate,
    policy_always_solve,
    policy_document_guided,
    policy_threshold_escalate,
)


artifacts_root = "Simulation_4/artifacts"
N_EPISODES = 1000
SEED_OFFSET = 70000
MODEL_PATH = "Simulation_4/artifacts/phase10_final/models/best_model"
MASKED_EVAL_PATH = "Simulation_4/artifacts/phase10_final/masked_evaluation.json"
CLEAN_EVAL_PATH = "Simulation_4/artifacts/phase10_final/clean_evaluation.json"


def run_policy_episodes(policy_fn, n: int = N_EPISODES, seed_offset: int = SEED_OFFSET):
    env = SupportEnv(artifacts_root=artifacts_root, nlg_enabled=False)
    rewards, resolutions, escalations, dropouts, timeouts, turns = [], [], [], [], [], []

    for ep in range(n):
        obs, info = env.reset(seed=seed_offset + ep)
        done = False
        ep_reward = 0.0

        while not done:
            action = policy_fn(obs, env.state)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            if truncated:
                break

        rewards.append(ep_reward)
        tt = info.get("last_transition_outcome", {}).get("terminal_type", "timeout")
        resolutions.append(1 if tt == "success" else 0)
        escalations.append(1 if tt == "escalation" else 0)
        dropouts.append(1 if tt == "dropout" else 0)
        timeouts.append(1 if tt == "timeout" else 0)
        turns.append(env.state["turn_count"])

    env.close()

    return {
        "mean_reward": float(np.mean(rewards)),
        "resolution_rate": float(np.mean(resolutions)),
        "escalation_rate": float(np.mean(escalations)),
        "dropout_rate": float(np.mean(dropouts)),
        "timeout_rate": float(np.mean(timeouts)),
        "mean_turns": float(np.mean(turns)),
    }


def run_ppo_episodes(model_path: str, n: int = N_EPISODES, seed_offset: int = SEED_OFFSET):
    model = PPO.load(model_path)
    env = SupportEnv(artifacts_root=artifacts_root, nlg_enabled=False)
    rewards, resolutions, escalations, dropouts, timeouts, turns = [], [], [], [], [], []
    esc_by_tier = {"Free": 0, "Pro": 0, "Business+": 0, "Enterprise": 0}
    tier_counts = {"Free": 0, "Pro": 0, "Business+": 0, "Enterprise": 0}

    for ep in range(n):
        obs, info = env.reset(seed=seed_offset + ep)
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            ep_reward += reward
            if truncated:
                break

        rewards.append(ep_reward)
        tt = info.get("last_transition_outcome", {}).get("terminal_type", "timeout")
        tier = env.state.get("tier", "Free")
        if tier == "Business":
            tier = "Business+"

        resolutions.append(1 if tt == "success" else 0)
        escalations.append(1 if tt == "escalation" else 0)
        dropouts.append(1 if tt == "dropout" else 0)
        timeouts.append(1 if tt == "timeout" else 0)
        turns.append(env.state["turn_count"])

        if tier in tier_counts:
            tier_counts[tier] += 1
        if tt == "escalation" and tier in esc_by_tier:
            esc_by_tier[tier] += 1

    env.close()

    result = {
        "mean_reward": float(np.mean(rewards)),
        "resolution_rate": float(np.mean(resolutions)),
        "escalation_rate": float(np.mean(escalations)),
        "dropout_rate": float(np.mean(dropouts)),
        "timeout_rate": float(np.mean(timeouts)),
        "mean_turns": float(np.mean(turns)),
        "escalation_by_tier": {
            tier: float(esc_by_tier[tier] / max(tier_counts[tier], 1)) for tier in esc_by_tier
        },
    }
    return result


def _normalize_tier(tier: str) -> str:
    if tier == "Business":
        return "Business+"
    return tier


def run_ppo_masked(model_path: str, n: int = N_EPISODES, seed_offset: int = SEED_OFFSET):
    model = PPO.load(model_path)
    env = ActionMaskedEnv(SupportEnv(artifacts_root=artifacts_root, nlg_enabled=False))

    rewards, resolutions, escalations, dropouts, timeouts, turns = [], [], [], [], [], []
    esc_by_tier = {"Free": 0, "Pro": 0, "Business+": 0, "Enterprise": 0}
    tier_counts = {"Free": 0, "Pro": 0, "Business+": 0, "Enterprise": 0}

    for ep in range(n):
        obs, info = env.reset(seed=seed_offset + ep)
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            mask = env.action_masks()
            if action < 0 or action >= len(mask) or not bool(mask[action]):
                action = 0

            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            if truncated:
                done = True

        rewards.append(ep_reward)
        tt = info.get("last_transition_outcome", {}).get("terminal_type", "timeout")
        tier = _normalize_tier(env.env.state.get("tier", "Free"))

        resolutions.append(1 if tt == "success" else 0)
        escalations.append(1 if tt == "escalation" else 0)
        dropouts.append(1 if tt == "dropout" else 0)
        timeouts.append(1 if tt == "timeout" else 0)
        turns.append(env.env.state["turn_count"])

        if tier in tier_counts:
            tier_counts[tier] += 1
        if tt == "escalation" and tier in esc_by_tier:
            esc_by_tier[tier] += 1

    env.close()

    return {
        "mean_reward": float(np.mean(rewards)),
        "resolution_rate": float(np.mean(resolutions)),
        "escalation_rate": float(np.mean(escalations)),
        "dropout_rate": float(np.mean(dropouts)),
        "timeout_rate": float(np.mean(timeouts)),
        "mean_turns": float(np.mean(turns)),
        "escalation_by_tier": {
            tier: float(esc_by_tier[tier] / max(tier_counts[tier], 1)) for tier in esc_by_tier
        },
    }


def run_baseline_masked(policy_fn, n: int = N_EPISODES, seed_offset: int = SEED_OFFSET):
    env = ActionMaskedEnv(SupportEnv(artifacts_root=artifacts_root, nlg_enabled=False))
    rewards, resolutions, escalations, dropouts, timeouts, turns = [], [], [], [], [], []

    for ep in range(n):
        obs, info = env.reset(seed=seed_offset + ep)
        done = False
        ep_reward = 0.0

        while not done:
            action = int(policy_fn(obs, env.env.state))
            mask = env.action_masks()
            if action < 0 or action >= len(mask) or not bool(mask[action]):
                action = 0

            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            if truncated:
                done = True

        rewards.append(ep_reward)
        tt = info.get("last_transition_outcome", {}).get("terminal_type", "timeout")
        resolutions.append(1 if tt == "success" else 0)
        escalations.append(1 if tt == "escalation" else 0)
        dropouts.append(1 if tt == "dropout" else 0)
        timeouts.append(1 if tt == "timeout" else 0)
        turns.append(env.env.state["turn_count"])

    env.close()

    return {
        "mean_reward": float(np.mean(rewards)),
        "resolution_rate": float(np.mean(resolutions)),
        "escalation_rate": float(np.mean(escalations)),
        "dropout_rate": float(np.mean(dropouts)),
        "timeout_rate": float(np.mean(timeouts)),
        "mean_turns": float(np.mean(turns)),
    }


def run_masked_eval(model_path: str) -> dict:
    results = {}
    print("PPO v3 (masked)...")
    results["PPO v3"] = run_ppo_masked(model_path)

    baselines = {
        "document_guided": policy_document_guided,
        "threshold_escalate": policy_threshold_escalate,
        "random": None,
        "always_solve": policy_always_solve,
        "always_escalate": policy_always_escalate,
    }

    for name, fn in baselines.items():
        print(f"{name} (masked)...")
        rng = np.random.default_rng(42)
        if name == "random":
            results[name] = run_baseline_masked(lambda obs, state, _rng=rng: int(_rng.integers(0, 5)))
        else:
            results[name] = run_baseline_masked(fn)

    return results


print("Running dual evaluation: masked fair comparison + unmasked robustness test...\n")

masked_results = run_masked_eval(MODEL_PATH)

unmasked_results = {}
print("PPO v3 (unmasked)...")
unmasked_results["PPO v3"] = run_ppo_episodes(MODEL_PATH)

baselines = {
    "document_guided": policy_document_guided,
    "threshold_escalate": policy_threshold_escalate,
    "random": None,
    "always_solve": policy_always_solve,
    "always_escalate": policy_always_escalate,
}

for name, fn in baselines.items():
    print(f"{name} (unmasked)...")
    rng = np.random.default_rng(42)
    if name == "random":
        unmasked_results[name] = run_policy_episodes(lambda obs, state, _rng=rng: int(_rng.integers(0, 5)))
    else:
        unmasked_results[name] = run_policy_episodes(fn)

print("\n=== EVALUATION A: MASKED (training conditions) ===")
print(f"{'Policy':<22} {'Reward':>8} {'Resolve':>8} {'Escalate':>9} {'Dropout':>8} {'Turns':>7}")
for name, res in sorted(masked_results.items(), key=lambda x: x[1]["mean_reward"], reverse=True):
    print(
        f"{name:<22} {res['mean_reward']:>8.3f} {res['resolution_rate']:>8.1%} "
        f"{res['escalation_rate']:>9.1%} {res['dropout_rate']:>8.1%} {res['mean_turns']:>7.1f}"
    )

print("\n=== EVALUATION B: UNMASKED (deployment robustness) ===")
print(f"{'Policy':<22} {'Reward':>8} {'Resolve':>8} {'Escalate':>9} {'Dropout':>8} {'Turns':>7}")
for name, res in sorted(unmasked_results.items(), key=lambda x: x[1]["mean_reward"], reverse=True):
    print(
        f"{name:<22} {res['mean_reward']:>8.3f} {res['resolution_rate']:>8.1%} "
        f"{res['escalation_rate']:>9.1%} {res['dropout_rate']:>8.1%} {res['mean_turns']:>7.1f}"
    )

mask_ppo = masked_results["PPO v3"]
unmask_ppo = unmasked_results["PPO v3"]
mask_doc = masked_results["document_guided"]
unmask_doc = unmasked_results["document_guided"]

print("\nKey findings:")
print(
    f"- Masked resolution rate: PPO vs document_guided = "
    f"{mask_ppo['resolution_rate']:.1%} vs {mask_doc['resolution_rate']:.1%}"
)
print(
    f"- Unmasked resolution rate: PPO vs document_guided = "
    f"{unmask_ppo['resolution_rate']:.1%} vs {unmask_doc['resolution_rate']:.1%}"
)
print(
    f"- Gap masked->unmasked for PPO: "
    f"{(unmask_ppo['resolution_rate'] - mask_ppo['resolution_rate']):+.1%}"
)
print(
    f"- Gap masked->unmasked for document_guided: "
    f"{(unmask_doc['resolution_rate'] - mask_doc['resolution_rate']):+.1%}"
)

Path(MASKED_EVAL_PATH).write_text(
    json.dumps(masked_results, indent=2)
)
Path(CLEAN_EVAL_PATH).write_text(
    json.dumps(unmasked_results, indent=2)
)
print(f"\nSaved: {MASKED_EVAL_PATH}")
print(f"Saved: {CLEAN_EVAL_PATH}")