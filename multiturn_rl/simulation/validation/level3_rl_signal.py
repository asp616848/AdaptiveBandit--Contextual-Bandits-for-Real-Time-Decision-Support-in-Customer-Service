from __future__ import annotations

from typing import Any

import numpy as np

from simulation.validation.baseline_policies import POLICY_REGISTRY
from simulation.validation.init import make_env, run_episode, safe_check


EXPECTED_POLICY_RANKING = [
    ("document_guided", 1),
    ("threshold_escalate", 2),
    ("random", 3),
    ("always_solve", 4),
    ("always_escalate", 5),
]


def run_level3(artifacts_root: str, n_episodes: int = 3000) -> dict[str, dict[str, Any]]:
    _ = artifacts_root
    env = make_env(nlg_enabled=False)

    seeds = list(range(20000, 20000 + n_episodes))
    policy_names = [name for name, _ in EXPECTED_POLICY_RANKING]

    metrics: dict[str, dict[str, Any]] = {}

    for p_name in policy_names:
        print(f"  [Level3] policy={p_name}")
        returns = []
        outcomes = {"resolved": 0, "escalated": 0, "dropout": 0, "timeout": 0}
        turns_to_resolve = []
        by_tier_escalation_rewards: dict[str, list[float]] = {"Free": [], "Enterprise": []}

        policy_fn = POLICY_REGISTRY[p_name]
        for idx, seed in enumerate(seeds):
            if idx % 500 == 0:
                print(f"    episode {idx}/{n_episodes}")
            rng = np.random.default_rng(seed + 170000)
            ep = run_episode(env, policy_fn, seed=seed, rng=rng)
            returns.append(float(ep["total_reward"]))
            outcomes[ep["terminal_type"]] += 1
            if ep["terminal_type"] == "resolved":
                turns_to_resolve.append(int(ep["turn_count"]))
            if ep["terminal_type"] == "escalated":
                tier = str(ep["state"].get("tier"))
                if tier in by_tier_escalation_rewards:
                    by_tier_escalation_rewards[tier].append(float(ep["total_reward"]))

        n = float(max(len(returns), 1))
        metrics[p_name] = {
            "mean_episode_reward": float(np.mean(returns)) if returns else 0.0,
            "std_episode_reward": float(np.std(returns)) if returns else 0.0,
            "resolution_rate": outcomes["resolved"] / n,
            "escalation_rate": outcomes["escalated"] / n,
            "dropout_rate": outcomes["dropout"] / n,
            "timeout_rate": outcomes["timeout"] / n,
            "mean_turns_to_resolution": float(np.mean(turns_to_resolve)) if turns_to_resolve else None,
            "n_escalations_free": int(len(by_tier_escalation_rewards["Free"])),
            "n_escalations_enterprise": int(len(by_tier_escalation_rewards["Enterprise"])),
            "mean_escalation_reward_free": float(np.mean(by_tier_escalation_rewards["Free"])) if by_tier_escalation_rewards["Free"] else None,
            "mean_escalation_reward_enterprise": float(np.mean(by_tier_escalation_rewards["Enterprise"])) if by_tier_escalation_rewards["Enterprise"] else None,
        }

    sorted_by_reward = sorted(policy_names, key=lambda n: metrics[n]["mean_episode_reward"], reverse=True)

    results: dict[str, dict[str, Any]] = {}

    def check_ranking_extremes() -> dict[str, Any]:
        expected_order = [x[0] for x in EXPECTED_POLICY_RANKING]
        ranking_ok = sorted_by_reward == expected_order
        return {
            "passed": ranking_ok,
            "value": {
                "sorted_by_reward": sorted_by_reward,
                "expected_order": expected_order,
                "mean_rewards": {k: metrics[k]["mean_episode_reward"] for k in policy_names},
            },
            "threshold": "policy ordering should match expected economic ordering",
        }

    def check_reward_signal_strength() -> dict[str, Any]:
        doc = float(metrics["document_guided"]["mean_episode_reward"])
        esc = float(metrics["always_escalate"]["mean_episode_reward"])
        gap = doc - esc
        passed = (doc > 0.0) and (esc < -1.0) and (gap > 2.0)
        return {
            "passed": passed,
            "value": {
                "document_guided_mean_reward": doc,
                "always_escalate_mean_reward": esc,
                "reward_gap": gap,
            },
            "threshold": "document_guided>0, always_escalate<-1.0, gap>2.0",
        }

    def check_resolution_vs_always_solve() -> dict[str, Any]:
        doc_rr = float(metrics["document_guided"]["resolution_rate"])
        solve_rr = float(metrics["always_solve"]["resolution_rate"])
        passed = doc_rr > solve_rr
        return {
            "passed": passed,
            "value": {
                "document_guided_resolution_rate": doc_rr,
                "always_solve_resolution_rate": solve_rr,
            },
            "threshold": "document_guided resolution_rate > always_solve resolution_rate",
        }

    def check_random_negative_reward() -> dict[str, Any]:
        random_mean = float(metrics["random"]["mean_episode_reward"])
        return {
            "passed": random_mean < 0.0,
            "value": {"random_mean_episode_reward": random_mean},
            "threshold": "random mean reward < 0",
        }

    def check_document_guided_profitability() -> dict[str, Any]:
        doc_mean = float(metrics["document_guided"]["mean_episode_reward"])
        return {
            "passed": doc_mean > 0.0,
            "value": {
                "document_guided_mean_episode_reward": doc_mean,
            },
            "threshold": "document_guided mean reward > 0",
        }

    results["check1_policy_ordering"] = safe_check("check1_policy_ordering", check_ranking_extremes)
    results["check2_reward_signal_strength"] = safe_check("check2_reward_signal_strength", check_reward_signal_strength)
    results["check3_resolution_doc_vs_always_solve"] = safe_check("check3_resolution_doc_vs_always_solve", check_resolution_vs_always_solve)
    results["check4_random_negative_reward"] = safe_check("check4_random_negative_reward", check_random_negative_reward)
    results["check5_document_guided_profitability"] = safe_check("check5_document_guided_profitability", check_document_guided_profitability)

    results["policy_metrics"] = {
        "passed": all(v.get("passed", False) for k, v in results.items() if k.startswith("check")),
        "value": metrics,
        "threshold": "informational",
    }

    return results
