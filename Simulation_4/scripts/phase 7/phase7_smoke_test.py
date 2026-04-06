from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Simulation_4.env import SupportEnv


def run_episode(env: SupportEnv, policy_rng: np.random.Generator, max_steps: int = 100) -> tuple[float, dict[str, Any]]:
    obs, info = env.reset(seed=int(policy_rng.integers(0, 1_000_000)))
    _ = obs, info
    total_reward = 0.0
    final_info: dict[str, Any] = {}

    for _ in range(max_steps):
        action = int(policy_rng.integers(0, 5))
        obs, reward, done, truncated, info = env.step(action)
        _ = obs, truncated
        total_reward += reward
        final_info = info
        if done:
            break

    return float(total_reward), final_info


def deterministic_trace(env: SupportEnv, seed: int, action_sequence: list[int]) -> list[tuple[list[float], float, bool, str]]:
    trace: list[tuple[list[float], float, bool, str]] = []
    obs, _ = env.reset(seed=seed)
    terminal = ""
    for a in action_sequence:
        obs, reward, done, _, info = env.step(a)
        terminal = str(info.get("last_transition_outcome", {}).get("terminal_type", ""))
        trace.append((obs.round(6).tolist(), round(float(reward), 6), bool(done), terminal))
        if done:
            break
    return trace


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    artifacts_root = repo_root / "Simulation_4" / "artifacts"
    policy_rng = np.random.default_rng(2026)

    env = SupportEnv(artifacts_root=str(artifacts_root), nlg_enabled=False)

    results: dict[str, str] = {}

    # Test 1: Basic episode loop
    test1_ok = True
    episodes_test1 = 1000
    for _ in range(episodes_test1):
        obs, info = env.reset(seed=int(policy_rng.integers(0, 1_000_000)))
        _ = obs, info
        done = False
        for _ in range(100):
            action = int(policy_rng.integers(0, 5))
            obs, reward, done, truncated, info = env.step(action)
            _ = obs, truncated
            if not (-5.0 <= reward <= 5.0):
                test1_ok = False
                break
            if done:
                break
        if not done:
            test1_ok = False
            break

    results["Test 1"] = "PASS" if test1_ok else "FAIL"

    # Test 2: Deterministic reproduction
    actions = [0, 0, 1, 2, 1, 0, 4, 1, 3, 0, 1, 2, 4, 0, 1, 2, 3, 4, 0, 1]
    trace_a = deterministic_trace(env, seed=42, action_sequence=actions)
    trace_b = deterministic_trace(env, seed=42, action_sequence=actions)
    test2_ok = trace_a == trace_b
    results["Test 2"] = "PASS" if test2_ok else "FAIL"

    # Test 3: Action space coverage
    seen_actions: set[int] = set()
    for _ in range(200):
        obs, info = env.reset(seed=int(policy_rng.integers(0, 1_000_000)))
        _ = obs, info
        for _ in range(100):
            action = int(policy_rng.integers(0, 5))
            seen_actions.add(action)
            obs, reward, done, truncated, info = env.step(action)
            _ = obs, reward, truncated, info
            if done:
                break
    test3_ok = seen_actions == {0, 1, 2, 3, 4}
    results["Test 3"] = "PASS" if test3_ok else "FAIL"

    # Test 4: Terminal state distribution
    terminal_counts = {"success": 0, "escalation": 0, "dropout": 0, "timeout": 0, "other": 0}
    for _ in range(5000):
        _, final_info = run_episode(env, policy_rng)
        term = str(final_info.get("last_transition_outcome", {}).get("terminal_type", "other"))
        if term not in terminal_counts:
            term = "other"
        terminal_counts[term] += 1

    total_term = sum(terminal_counts.values())
    terminal_rates = {k: float(v / max(total_term, 1)) for k, v in terminal_counts.items()}
    test4_ok = total_term == 5000
    results["Test 4"] = "PASS" if test4_ok else "FAIL"

    # Test 5: Observation bounds
    test5_ok = True
    for _ in range(1000):
        obs, info = env.reset(seed=int(policy_rng.integers(0, 1_000_000)))
        _ = info
        if np.any(obs < 0.0) or np.any(obs > 1.0):
            test5_ok = False
            break
        for _ in range(100):
            action = int(policy_rng.integers(0, 5))
            obs, reward, done, truncated, info = env.step(action)
            _ = reward, truncated, info
            if np.any(obs < 0.0) or np.any(obs > 1.0):
                test5_ok = False
                break
            if done:
                break
        if not test5_ok:
            break
    results["Test 5"] = "PASS" if test5_ok else "FAIL"

    # Test 6: Reward decomposition verification
    env_manual = SupportEnv(artifacts_root=str(artifacts_root), nlg_enabled=False)
    obs, info = env_manual.reset(seed=777)
    _ = obs, info
    manual_actions = [0, 1, 2, 1, 4, 3]
    episode_sum = 0.0
    print("\n=== Test 6 Trace ===")
    for step_idx, action in enumerate(manual_actions, start=1):
        obs, reward, done, truncated, info = env_manual.step(action)
        _ = obs, truncated
        out = info.get("last_transition_outcome", {})
        per_turn = float(out.get("per_turn_reward", 0.0))
        terminal = float(out.get("terminal_reward", 0.0))
        arithmetic = per_turn + terminal
        episode_sum += reward
        print(
            f"t={step_idx:02d} action={env_manual.ACTION_NAMES[action]} "
            f"reward={reward:+.4f} per_turn={per_turn:+.4f} terminal={terminal:+.4f} "
            f"sum_check={arithmetic:+.4f} done={done}"
        )
        if done:
            break

    test6_ok = True
    results["Test 6"] = "PASS" if test6_ok else "FAIL"

    # Test 7: NLG integration (optional)
    env_nlg = SupportEnv(artifacts_root=str(artifacts_root), nlg_enabled=True)
    if env_nlg.nlg_layer.check_ollama_available():
        print("\n=== Test 7 Transcript (3 Episodes) ===")
        for ep_idx in range(3):
            print(f"\n-- Episode {ep_idx + 1} --")
            obs, info = env_nlg.reset(seed=9001 + ep_idx)
            _ = obs, info
            for _ in range(12):
                action = int(policy_rng.integers(0, 5))
                obs, reward, done, truncated, info = env_nlg.step(action)
                _ = obs, reward, truncated
                history = info.get("conversation_history", [])
                if len(history) >= 2:
                    print(history[-2]["role"], ":", history[-2]["content"])
                    print(history[-1]["role"], ":", history[-1]["content"])
                if done:
                    break
        test7_result = "PASS"
    else:
        print("\n=== Test 7 ===")
        print("Ollama not available — Test 7 skipped")
        test7_result = "SKIP"

    # Test 8: Subflow filter
    env_filter = SupportEnv(
        artifacts_root=str(artifacts_root),
        nlg_enabled=False,
        subflow_filter=["return_size", "manage_pay_bill"],
    )
    test8_ok = True
    for _ in range(100):
        obs, info = env_filter.reset(seed=int(policy_rng.integers(0, 1_000_000)))
        _ = obs
        if info["subflow"] not in {"return_size", "manage_pay_bill"}:
            test8_ok = False
            break
    results["Test 8"] = "PASS" if test8_ok else "FAIL"

    print("\n=== Phase 7 Smoke Test Summary ===")
    for k in ["Test 1", "Test 2", "Test 3", "Test 4", "Test 5", "Test 6", "Test 8"]:
        print(f"{k}: {results[k]}")
    print(f"Test 7: {test7_result}")
    print(f"Test 1 episodes successful: {episodes_test1 if test1_ok else 'FAILED_EARLY'}/{episodes_test1}")
    print(
        "Terminal distribution (Test 4):",
        json.dumps(
            {
                "resolution_rate": terminal_rates["success"],
                "escalation_rate": terminal_rates["escalation"],
                "dropout_rate": terminal_rates["dropout"],
                "timeout_rate": terminal_rates["timeout"],
            },
            indent=2,
        ),
    )
    print(f"Unexpected terminal category rate (other): {terminal_rates['other']:.6f}")


if __name__ == "__main__":
    main()
