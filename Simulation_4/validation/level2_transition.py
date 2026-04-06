from __future__ import annotations

from typing import Any

import numpy as np

from Simulation_4.validation.baseline_policies import policy_random
from Simulation_4.validation.init import make_env, run_episode, safe_check


def run_level2(artifacts_root: str, n_episodes: int = 2000) -> dict[str, dict[str, Any]]:
    _ = artifacts_root
    env = make_env(nlg_enabled=False)

    ask_gain = []
    ask_nogain = []
    ask_gain_flags = []

    ps_fail_delta_f = []
    ps_success_delta_f = []
    ps_fail_streak_delta = []
    ps_success_streak_post = []

    repair_eff_delta_f = []
    repair_ineff_delta_f = []
    repair_effective_flags = []
    repair_rho_values = []

    escalate_checks = []
    close_checks = []

    failed_streak_after_success = []

    pdrop_samples = []
    psuccess_low_info = []
    psuccess_high_info = []
    autoresolve_checks = []
    resolved_violation_flags = []

    for ep_idx, seed in enumerate(range(10000, 10000 + n_episodes)):
        if ep_idx % 500 == 0:
            print(f"  [Level2] episode {ep_idx}/{n_episodes}")

        rng = np.random.default_rng(seed + 130000)
        ep = run_episode(env, policy_random, seed=seed, rng=rng)

        for tr in ep["transitions"]:
            action = tr["action_name"]
            pre = tr["pre"]
            post = tr["post"]
            out = tr["outcome"]

            pre_f = float(pre.get("frustration", 0.0))
            post_f = float(post.get("frustration", 0.0))
            delta_f = post_f - pre_f

            pdrop_samples.append((float(pre.get("frustration", 0.0)), float(tr.get("pre_p_dropout", 0.0))))

            info_val = float(pre.get("information", 0.0))
            p_success = float(tr.get("pre_p_success", 0.0))
            if action == "ProvideSolution":
                if info_val < 0.3:
                    psuccess_low_info.append(p_success)
                if info_val > 0.6:
                    psuccess_high_info.append(p_success)

            if bool(post.get("resolved", 0)):
                triggered_by_close_success = (action == "Close") and bool(out.get("resolved", False))
                triggered_by_auto = (action == "ProvideSolution") and bool(out.get("outcome") == "success") and (float(post.get("progress", 0.0)) >= 0.85)
                resolved_violation_flags.append(triggered_by_close_success or triggered_by_auto)

            if action == "AskInfo":
                gained = bool(out.get("gain_occurred", False))
                delta_i = float(out.get("delta_i", 0.0))
                ask_gain_flags.append(gained)
                if gained:
                    ask_gain.append(delta_i)
                else:
                    ask_nogain.append(delta_i)

            elif action == "ProvideSolution":
                outcome = str(out.get("outcome", ""))
                if outcome == "failure":
                    ps_fail_delta_f.append(delta_f)
                    ps_fail_streak_delta.append(int(post.get("failed_streak", 0)) - int(pre.get("failed_streak", 0)))
                elif outcome == "success":
                    ps_success_delta_f.append(delta_f)
                    ps_success_streak_post.append(int(post.get("failed_streak", 0)))
                    failed_streak_after_success.append(int(post.get("failed_streak", 0)))

                    if float(post.get("progress", 0.0)) >= 0.85:
                        autoresolve_checks.append(bool(post.get("resolved", 0)))

            elif action == "AffectiveRepair":
                effective = bool(out.get("repair_effective", False))
                repair_effective_flags.append(effective)
                repair_rho_values.append(float(pre.get("rho", 0.0)))
                if effective:
                    repair_eff_delta_f.append(delta_f)
                else:
                    repair_ineff_delta_f.append(delta_f)

            elif action == "Escalate":
                escalate_checks.append(
                    bool(post.get("done", False))
                    and bool(post.get("escalated", 0))
                    and not bool(post.get("resolved", 0))
                )

            elif action == "Close":
                resolved = bool(out.get("resolved", False))
                if resolved:
                    close_checks.append(bool(post.get("done", False)) and bool(post.get("resolved", 0)))
                else:
                    close_checks.append(bool(post.get("done", False)) and not bool(post.get("resolved", 0)))

    results: dict[str, dict[str, Any]] = {}

    def check_askinfo_gain() -> dict[str, Any]:
        mean_gain = float(np.mean(ask_gain)) if ask_gain else 0.0
        max_no_gain = float(np.max(np.abs(ask_nogain))) if ask_nogain else 0.0
        p_gain = float(np.mean(ask_gain_flags)) if ask_gain_flags else 0.0
        info_non_decreasing = all(delta >= -1e-10 for delta in (ask_gain + ask_nogain))
        passed = (mean_gain > 0.10) and (max_no_gain == 0.0) and (0.15 <= p_gain <= 0.45) and info_non_decreasing
        return {
            "passed": passed,
            "value": {
                "mean_delta_i_when_gain": mean_gain,
                "max_abs_delta_i_when_no_gain": max_no_gain,
                "p_gain_occurred": p_gain,
                "delta_i_non_negative": info_non_decreasing,
            },
            "threshold": "mean_delta_i(gain)>0.10, delta_i(no_gain)==0, 0.15<=p_gain<=0.45, delta_i>=0",
        }

    def check_provide_solution_dynamics() -> dict[str, Any]:
        mean_fail_delta = float(np.mean(ps_fail_delta_f)) if ps_fail_delta_f else 0.0
        mean_succ_delta = float(np.mean(ps_success_delta_f)) if ps_success_delta_f else 0.0
        streak_fail_ok = all(x == 1 for x in ps_fail_streak_delta) if ps_fail_streak_delta else False
        streak_succ_ok = all(x == 0 for x in ps_success_streak_post) if ps_success_streak_post else False
        passed = (mean_fail_delta > 0.0) and (mean_succ_delta < 0.0) and streak_fail_ok and streak_succ_ok
        return {
            "passed": passed,
            "value": {
                "mean_delta_f_failure": mean_fail_delta,
                "mean_delta_f_success": mean_succ_delta,
                "failed_streak_increment_exactly_1_on_failure": streak_fail_ok,
                "failed_streak_reset_0_on_success": streak_succ_ok,
            },
            "threshold": "failure delta_f>0, success delta_f<0, streak +1 on failure, streak 0 on success",
        }

    def check_affective_repair() -> dict[str, Any]:
        mean_eff = float(np.mean(repair_eff_delta_f)) if repair_eff_delta_f else 0.0
        mean_ineff = float(np.mean(repair_ineff_delta_f)) if repair_ineff_delta_f else 0.0
        p_eff = float(np.mean(repair_effective_flags)) if repair_effective_flags else 0.0

        low_idx = [i for i, rho in enumerate(repair_rho_values) if rho < 0.4]
        high_idx = [i for i, rho in enumerate(repair_rho_values) if rho > 0.6]
        low_rate = float(np.mean([repair_effective_flags[i] for i in low_idx])) if low_idx else 0.0
        high_rate = float(np.mean([repair_effective_flags[i] for i in high_idx])) if high_idx else 0.0

        passed = (mean_eff < 0.0) and (mean_ineff >= 0.0) and (0.35 <= p_eff <= 0.85) and (high_rate > low_rate)
        return {
            "passed": passed,
            "value": {
                "mean_delta_f_effective": mean_eff,
                "mean_delta_f_ineffective": mean_ineff,
                "p_repair_effective": p_eff,
                "high_rho_effective_rate": high_rate,
                "low_rho_effective_rate": low_rate,
            },
            "threshold": "effective delta_f<0, ineffective delta_f>=0, 0.35<=p_effective<=0.85, high_rho_rate>low_rho_rate",
        }

    def check_terminal_actions() -> dict[str, Any]:
        esc_ok = all(escalate_checks) if escalate_checks else False
        close_ok = all(close_checks) if close_checks else False
        return {
            "passed": esc_ok and close_ok,
            "value": {
                "escalate_terminal_assertions": esc_ok,
                "close_terminal_assertions": close_ok,
                "n_escalate": len(escalate_checks),
                "n_close": len(close_checks),
            },
            "threshold": "All Escalate/Close actions terminal with expected flags",
        }

    def check_failed_streak_reset() -> dict[str, Any]:
        reset_ok = all(x == 0 for x in failed_streak_after_success) if failed_streak_after_success else False
        return {
            "passed": reset_ok,
            "value": {
                "n_successful_provide_solution": len(failed_streak_after_success),
                "all_reset_to_zero": reset_ok,
            },
            "threshold": "failed_streak==0 immediately after successful ProvideSolution",
        }

    def check_dropout_monotone() -> dict[str, Any]:
        if not pdrop_samples:
            raise RuntimeError("No p_dropout samples collected")

        fr = np.array([x[0] for x in pdrop_samples], dtype=float)
        pd = np.array([x[1] for x in pdrop_samples], dtype=float)

        b08 = pd[fr >= 0.8]
        b04 = pd[(fr >= 0.35) & (fr <= 0.45)]
        b01 = pd[(fr >= 0.05) & (fr <= 0.15)]
        if len(b08) == 0 or len(b04) == 0 or len(b01) == 0:
            raise RuntimeError("Insufficient p_dropout samples in one or more frustration bins")

        mean08 = float(np.mean(b08))
        mean04 = float(np.mean(b04))
        mean01 = float(np.mean(b01))
        corr = float(np.corrcoef(fr, pd)[0, 1]) if len(fr) > 1 else 0.0

        passed = (mean08 > mean04 > mean01) and (corr > 0.5)
        return {
            "passed": passed,
            "value": {
                "mean_pdrop_f08": mean08,
                "mean_pdrop_f04": mean04,
                "mean_pdrop_f01": mean01,
                "corr_frustration_pdrop": corr,
            },
            "threshold": "pdrop(0.8)>pdrop(0.4)>pdrop(0.1), corr(frustration,pdrop)>0.5",
        }

    def check_psuccess_increases_with_information() -> dict[str, Any]:
        if not psuccess_low_info or not psuccess_high_info:
            raise RuntimeError("Insufficient ProvideSolution transitions in low/high information bins")

        mean_low = float(np.mean(psuccess_low_info))
        mean_high = float(np.mean(psuccess_high_info))
        passed = mean_high > mean_low
        return {
            "passed": passed,
            "value": {
                "mean_psuccess_info_lt_0_3": mean_low,
                "mean_psuccess_info_gt_0_6": mean_high,
                "n_low_info_samples": int(len(psuccess_low_info)),
                "n_high_info_samples": int(len(psuccess_high_info)),
            },
            "threshold": "mean p_success at info>0.6 should exceed mean p_success at info<0.3",
        }

    def check_autoresolve_trigger() -> dict[str, Any]:
        if not autoresolve_checks:
            raise RuntimeError("No successful ProvideSolution transitions with post_progress>=0.85 observed")
        all_autoresolved = all(autoresolve_checks)
        no_spurious_resolve = all(resolved_violation_flags) if resolved_violation_flags else True
        passed = all_autoresolved and no_spurious_resolve
        return {
            "passed": passed,
            "value": {
                "all_autoresolve_successes_resolved": all_autoresolved,
                "resolved_only_when_allowed": no_spurious_resolve,
                "n_autoresolve_candidates": len(autoresolve_checks),
            },
            "threshold": "ProvideSolution success with post_progress>=0.85 always resolves; no spurious resolved states",
        }

    results["check1_askinfo_gain"] = safe_check("check1_askinfo_gain", check_askinfo_gain)
    results["check2_provide_solution_dynamics"] = safe_check("check2_provide_solution_dynamics", check_provide_solution_dynamics)
    results["check3_affective_repair"] = safe_check("check3_affective_repair", check_affective_repair)
    results["check4_escalate_close_terminal"] = safe_check("check4_escalate_close_terminal", check_terminal_actions)
    results["check5_failed_streak_reset"] = safe_check("check5_failed_streak_reset", check_failed_streak_reset)
    results["check6_dropout_monotone"] = safe_check("check6_dropout_monotone", check_dropout_monotone)
    results["check7_psuccess_vs_information"] = safe_check("check7_psuccess_vs_information", check_psuccess_increases_with_information)
    results["check8_autoresolve"] = safe_check("check8_autoresolve", check_autoresolve_trigger)

    return results
