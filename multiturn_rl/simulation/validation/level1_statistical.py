from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

import numpy as np

from simulation.validation.baseline_policies import policy_document_guided
from simulation.validation.init import make_env, run_episode, safe_check


ABCD_BENCHMARKS = {
    "resolution_rate": 0.864,
    "escalation_rate": 0.059,
    "dropout_rate": 0.0076,
    "mean_turn_count": None,
    "resolution_rate_tolerance": 0.15,
    "escalation_rate_tolerance": 0.10,
}


def run_level1(artifacts_root: str, n_episodes: int = 5000) -> dict[str, dict[str, Any]]:
    _ = artifacts_root
    env = make_env(nlg_enabled=False)
    trajectories_i = {1: [], 3: [], 5: [], 7: [], 10: []}
    trajectories_f = {1: [], 3: [], 5: [], 7: [], 10: []}

    outcomes = defaultdict(int)
    turn_counts: list[int] = []
    all_info_values: list[float] = []
    repair_deltas: list[float] = []
    provide_fail_deltas: list[float] = []

    for ep in range(n_episodes):
        if ep % 500 == 0:
            print(f"  [Level1] episode {ep}/{n_episodes}")

        rng = np.random.default_rng(ep + 90000)
        ep_result = run_episode(env, policy_document_guided, seed=ep, rng=rng)

        outcomes[ep_result["terminal_type"]] += 1
        turn_counts.append(int(ep_result["turn_count"]))

        info_by_turn = [float(t["post"].get("information", 0.0)) for t in ep_result["transitions"]]
        fr_by_turn = [float(t["post"].get("frustration", 0.0)) for t in ep_result["transitions"]]
        all_info_values.extend(info_by_turn)

        if info_by_turn:
            for checkpoint in trajectories_i:
                idx = min(checkpoint, len(info_by_turn)) - 1
                trajectories_i[checkpoint].append(float(info_by_turn[idx]))
                trajectories_f[checkpoint].append(float(fr_by_turn[idx]))

        for tr in ep_result["transitions"]:
            action_name = tr["action_name"]
            pre_f = float(tr["pre"].get("frustration", 0.0))
            post_f = float(tr["post"].get("frustration", 0.0))
            delta_f = post_f - pre_f
            if action_name == "AffectiveRepair":
                repair_deltas.append(delta_f)
            if action_name == "ProvideSolution" and tr["outcome"].get("outcome") == "failure":
                provide_fail_deltas.append(delta_f)

    n = float(max(n_episodes, 1))
    resolution_rate = outcomes["resolved"] / n
    escalation_rate = outcomes["escalated"] / n
    dropout_rate = outcomes["dropout"] / n
    timeout_rate = outcomes["timeout"] / n

    mean_info = {k: float(np.mean(v)) if v else 0.0 for k, v in trajectories_i.items()}
    mean_fr = {k: float(np.mean(v)) if v else 0.0 for k, v in trajectories_f.items()}

    results: dict[str, dict[str, Any]] = {}

    def check_terminal_distribution() -> dict[str, Any]:
        total = resolution_rate + escalation_rate + dropout_rate + timeout_rate
        passed = (
            resolution_rate > 0.40
            and escalation_rate < 0.20
            and abs(total - 1.0) <= 0.001
        )
        return {
            "passed": passed,
            "value": {
                "resolution_rate": resolution_rate,
                "escalation_rate": escalation_rate,
                "dropout_rate": dropout_rate,
                "timeout_rate": timeout_rate,
                "sum": total,
            },
            "threshold": "resolution>0.40, escalation<0.20, sum within 1.0±0.001",
        }

    def check_turn_distribution() -> dict[str, Any]:
        mean_turn = float(np.mean(turn_counts)) if turn_counts else 0.0
        no_zero_endings = int(sum(1 for t in turn_counts if t == 0)) == 0
        long_episode_share = float(sum(1 for t in turn_counts if t >= 10)) / float(max(len(turn_counts), 1))
        passed = (5.0 <= mean_turn <= 18.0) and no_zero_endings and (long_episode_share >= 0.10)
        return {
            "passed": passed,
            "value": {
                "mean_turn_count": mean_turn,
                "zero_turn_episodes": int(sum(1 for t in turn_counts if t == 0)),
                "turn_ge_10_share": long_episode_share,
            },
            "threshold": "5<=mean_turn<=18, zero_turn_episodes==0, turn>=10 share>=0.10",
        }

    def check_information_trajectory() -> dict[str, Any]:
        checkpoints = sorted(mean_info.keys())
        monotone = all(mean_info[checkpoints[i + 1]] >= mean_info[checkpoints[i]] - 1e-8 for i in range(len(checkpoints) - 1))
        turn5_gt_turn1 = mean_info[5] > mean_info[1]
        bounded = (min(all_info_values) >= 0.0) and (max(all_info_values) <= 1.0) if all_info_values else True
        passed = monotone and turn5_gt_turn1 and bounded
        return {
            "passed": passed,
            "value": {
                "mean_information_by_turn": mean_info,
                "turn5_gt_turn1": turn5_gt_turn1,
                "bounded_0_1": bounded,
            },
            "threshold": "monotone non-decreasing means, mean_info_t5>mean_info_t1, information in [0,1]",
        }

    def check_frustration_trajectory() -> dict[str, Any]:
        mean_repair_delta = float(np.mean(repair_deltas)) if repair_deltas else 0.0
        mean_fail_delta = float(np.mean(provide_fail_deltas)) if provide_fail_deltas else 0.0
        turn10_ok = mean_fr[10] <= 0.50
        passed = turn10_ok and (mean_repair_delta < 0.0) and (mean_fail_delta > 0.0)
        return {
            "passed": passed,
            "value": {
                "mean_frustration_by_turn": mean_fr,
                "mean_delta_f_affective_repair": mean_repair_delta,
                "mean_delta_f_provide_solution_failure": mean_fail_delta,
            },
            "threshold": "mean_frustration_t10<=0.50, mean_delta_f(repair)<0, mean_delta_f(solution_failure)>0",
        }

    def check_persona_differentiation() -> dict[str, Any]:
        personas = [
            "high_engagement_resolver",
            "low_engagement_resolver",
            "silent_dropout",
            "escalation_prone",
        ]
        persona_counts = {p: 0 for p in personas}
        persona_outcomes = {p: defaultdict(int) for p in personas}

        seed = 50000
        max_attempts = 120000
        attempts = 0
        while attempts < max_attempts and min(persona_counts.values()) < 1000:
            attempts += 1
            rng_local = np.random.default_rng(seed + 91000)
            ep = run_episode(env, policy_document_guided, seed=seed, rng=rng_local)
            p = str(ep["state"].get("persona_label"))
            seed += 1
            if p not in persona_counts:
                continue
            if persona_counts[p] >= 1000:
                continue
            persona_counts[p] += 1
            persona_outcomes[p][ep["terminal_type"]] += 1

        if min(persona_counts.values()) < 1000:
            raise RuntimeError(f"Could not collect 1000 episodes for all personas. Counts={persona_counts}")

        table: dict[str, dict[str, float]] = {}
        for p in personas:
            total_p = float(max(persona_counts[p], 1))
            table[p] = {
                "resolution_rate": persona_outcomes[p]["resolved"] / total_p,
                "escalation_rate": persona_outcomes[p]["escalated"] / total_p,
                "dropout_rate": persona_outcomes[p]["dropout"] / total_p,
                "timeout_rate": persona_outcomes[p]["timeout"] / total_p,
            }

        cond1 = table["escalation_prone"]["escalation_rate"] > table["high_engagement_resolver"]["escalation_rate"]
        cond2 = table["silent_dropout"]["dropout_rate"] > table["high_engagement_resolver"]["dropout_rate"]
        best_resolution_persona = max(personas, key=lambda x: table[x]["resolution_rate"])
        cond3 = best_resolution_persona == "high_engagement_resolver"

        return {
            "passed": cond1 and cond2 and cond3,
            "value": {
                "persona_outcome_rates": table,
                "persona_counts": persona_counts,
                "best_resolution_persona": best_resolution_persona,
            },
            "threshold": "escalation_prone escalates more than high_engagement; silent_dropout drops out more than high_engagement; high_engagement has highest resolution",
        }

    def check_tier_reward_ordering() -> dict[str, Any]:
        tiers = ["Free", "Pro", "Business", "Enterprise"]
        samples_needed = 500
        tier_counts = {t: 0 for t in tiers}
        rewards_by_tier_outcome: dict[str, dict[str, list[float]]] = {
            t: {"resolved": [], "escalated": []} for t in tiers
        }

        seed = 80000
        max_attempts = 200000
        attempts = 0
        while attempts < max_attempts and min(tier_counts.values()) < samples_needed:
            attempts += 1
            rng_local = np.random.default_rng(seed + 92000)
            ep = run_episode(env, policy_document_guided, seed=seed, rng=rng_local)
            tier = str(ep["state"].get("tier"))
            seed += 1
            if tier not in tier_counts:
                continue
            if tier_counts[tier] >= samples_needed:
                continue

            tier_counts[tier] += 1
            if ep["terminal_type"] == "resolved":
                rewards_by_tier_outcome[tier]["resolved"].append(float(ep["total_reward"]))
            if ep["terminal_type"] == "escalated":
                rewards_by_tier_outcome[tier]["escalated"].append(float(ep["total_reward"]))

        if min(tier_counts.values()) < samples_needed:
            raise RuntimeError(f"Could not collect 500 episodes for all tiers. Counts={tier_counts}")

        mean_rewards: dict[str, dict[str, float | None]] = {}
        for t in tiers:
            mean_rewards[t] = {
                "resolved": float(np.mean(rewards_by_tier_outcome[t]["resolved"])) if rewards_by_tier_outcome[t]["resolved"] else None,
                "escalated": float(np.mean(rewards_by_tier_outcome[t]["escalated"])) if rewards_by_tier_outcome[t]["escalated"] else None,
            }

        free_esc = mean_rewards["Free"]["escalated"]
        ent_esc = mean_rewards["Enterprise"]["escalated"]
        free_res = mean_rewards["Free"]["resolved"]
        ent_res = mean_rewards["Enterprise"]["resolved"]

        if free_esc is None or ent_esc is None or free_res is None or ent_res is None:
            raise RuntimeError(f"Missing resolved/escalated samples for some tiers: {mean_rewards}")

        cond1 = float(ent_esc) > float(free_esc)
        cond2 = True
        for t in tiers:
            if mean_rewards[t]["resolved"] is None or mean_rewards[t]["escalated"] is None:
                cond2 = False
                break
            cond2 = cond2 and (float(mean_rewards[t]["resolved"]) > float(mean_rewards[t]["escalated"]))
        cond3 = abs(float(free_res) - float(ent_res)) <= 0.75 and (3.0 <= float(free_res) <= 5.0) and (3.0 <= float(ent_res) <= 5.0)

        return {
            "passed": cond1 and cond2 and cond3,
            "value": {
                "mean_rewards_by_tier": mean_rewards,
                "tier_counts": tier_counts,
            },
            "threshold": "Enterprise escalation reward > Free escalation reward; resolved > escalated by tier; Free vs Enterprise resolved rewards approximately equal around +4",
        }

    results["check1_terminal_outcomes"] = safe_check("check1_terminal_outcomes", check_terminal_distribution)
    results["check2_turn_distribution"] = safe_check("check2_turn_distribution", check_turn_distribution)
    results["check3_information_trajectory"] = safe_check("check3_information_trajectory", check_information_trajectory)
    results["check4_frustration_trajectory"] = safe_check("check4_frustration_trajectory", check_frustration_trajectory)
    results["check5_persona_differentiation"] = safe_check("check5_persona_differentiation", check_persona_differentiation)
    results["check6_tier_reward_ordering"] = safe_check("check6_tier_reward_ordering", check_tier_reward_ordering)

    results["summary_level1_metrics"] = {
        "passed": all(v.get("passed", False) for k, v in results.items() if k.startswith("check")),
        "value": {
            "resolution_rate": resolution_rate,
            "escalation_rate": escalation_rate,
            "dropout_rate": dropout_rate,
            "timeout_rate": timeout_rate,
            "mean_turn_count": float(mean(turn_counts)) if turn_counts else 0.0,
        },
        "threshold": "informational",
    }

    return results
