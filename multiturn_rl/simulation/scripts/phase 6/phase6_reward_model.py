from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ChurnState:
    name: str
    frustration: float
    failed_streak: int
    turn_count: int
    tau: float
    expected_lo: float
    expected_hi: float


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-x)))


def locate_file(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of the candidate paths exist: {[str(c) for c in candidates]}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def calibrate_churn_coefficients(states: list[ChurnState]) -> dict[str, Any]:
    # Requested defaults before calibration.
    base = {"c0": -3.8, "cf": 4.2, "cs": 0.4, "ct": 0.08, "ctau": -1.5}

    def eval_with(coeffs: dict[str, float]) -> tuple[list[float], bool, float]:
        vals: list[float] = []
        penalty = 0.0
        monotonic = True
        for s in states:
            z = (
                coeffs["c0"]
                + coeffs["cf"] * s.frustration
                + coeffs["cs"] * s.failed_streak
                + coeffs["ct"] * s.turn_count
                + coeffs["ctau"] * s.tau
            )
            p = sigmoid(z)
            vals.append(p)
            center = 0.5 * (s.expected_lo + s.expected_hi)
            width = max(1e-6, s.expected_hi - s.expected_lo)
            penalty += ((p - center) / width) ** 2
        for i in range(len(vals) - 1):
            if vals[i + 1] < vals[i]:
                monotonic = False
                penalty += 10.0
        within = all(s.expected_lo <= p <= s.expected_hi for s, p in zip(states, vals))
        return vals, within and monotonic, penalty

    base_vals, base_ok, base_penalty = eval_with(base)

    best = {"coeffs": base, "vals": base_vals, "ok": base_ok, "penalty": base_penalty}

    # Calibrate only if defaults are outside target bands.
    if not base_ok:
        c0_grid = np.round(np.arange(-4.8, -3.6 + 1e-9, 0.1), 2)
        cf_grid = np.round(np.arange(3.0, 4.4 + 1e-9, 0.1), 2)
        cs_grid = np.round(np.arange(0.1, 0.6 + 1e-9, 0.1), 2)

        for c0 in c0_grid:
            for cf in cf_grid:
                for cs in cs_grid:
                    cand = {"c0": float(c0), "cf": float(cf), "cs": float(cs), "ct": 0.08, "ctau": -1.5}
                    vals, ok, penalty = eval_with(cand)
                    if ok:
                        if (not best["ok"]) or (penalty < best["penalty"]):
                            best = {"coeffs": cand, "vals": vals, "ok": ok, "penalty": penalty}
                    elif not best["ok"] and penalty < best["penalty"]:
                        best = {"coeffs": cand, "vals": vals, "ok": ok, "penalty": penalty}

    checks = []
    for s, p in zip(states, best["vals"]):
        checks.append(
            {
                "state": s.name,
                "frustration": s.frustration,
                "failed_streak": s.failed_streak,
                "turn_count": s.turn_count,
                "tau": s.tau,
                "p_churn": float(p),
                "expected_range": [s.expected_lo, s.expected_hi],
                "within_expected_range": bool(s.expected_lo <= p <= s.expected_hi),
            }
        )

    return {
        "defaults": base,
        "defaults_valid": bool(base_ok),
        "selected": best["coeffs"],
        "selected_valid": bool(best["ok"]),
        "state_checks": checks,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    sim_root = repo_root / "simulation"
    phase1_dir = sim_root / "artifacts" / "phase 1"
    phase6_dir = sim_root / "artifacts" / "phase 6"
    phase6_dir.mkdir(parents=True, exist_ok=True)

    extract1_path = locate_file([phase1_dir / "extract1_conversation_stats.csv"])
    extract3_path = locate_file([phase1_dir / "extract3_subflow_stats.csv"])
    phase2_path = locate_file([
        sim_root / "artifacts" / "phase 2" / "state_vector_spec.json",
        sim_root / "artifacts" / "phase2" / "state_vector_spec.json",
    ])
    phase3_path = locate_file([
        sim_root / "artifacts" / "phase 3" / "transition_calibration.json",
        sim_root / "artifacts" / "phase3" / "transition_calibration.json",
    ])
    phase5_path = locate_file([
        sim_root / "artifacts" / "phase 5" / "persona_profiles.json",
        sim_root / "artifacts" / "phase5" / "persona_profiles.json",
    ])

    extract1 = pd.read_csv(extract1_path)
    extract3 = pd.read_csv(extract3_path)
    state_spec = load_json(phase2_path)
    transition = load_json(phase3_path)
    personas = load_json(phase5_path)

    tier_freqs = (
        state_spec.get("static_variables", {})
        .get("tier", {})
        .get("init_frequencies", {})
    )

    tier_mapping = {
        "guest": "Free",
        "bronze": "Pro",
        "silver": "Business",
        "gold": "Enterprise",
    }
    excluded_member_levels = {
        "platinum": "Excluded from calibration: 0% init frequency in ABCD.",
        "vip": "Excluded from calibration: 0% init frequency in ABCD.",
    }

    business_tier_probs = {"Free": 0.0, "Pro": 0.0, "Business": 0.0, "Enterprise": 0.0}
    for lvl, p in tier_freqs.items():
        if lvl in tier_mapping:
            business_tier_probs[tier_mapping[lvl]] += float(p)

    # Normalize defensively if inputs do not sum to 1.
    total_p = sum(business_tier_probs.values())
    if total_p > 0:
        business_tier_probs = {k: float(v / total_p) for k, v in business_tier_probs.items()}

    base_value_by_tier = {
        "Free": 1.6,
        "Pro": 3.4,
        "Business": 7.5,
        "Enterprise": 12.0,
    }
    escalation_cost_by_tier = {
        "Free": 0.0,
        "Pro": 0.0,
        "Business": 1.0,
        "Enterprise": 2.0,
    }
    kappa = 0.5

    representative_value_weight = {"Free": 0.05, "Pro": 0.425, "Business": 0.75, "Enterprise": 1.0}

    expected_value_by_tier = {
        tier: float(base_value_by_tier[tier] * (1.0 + kappa * representative_value_weight[tier]))
        for tier in base_value_by_tier
    }

    avg_value = float(sum(business_tier_probs[t] * expected_value_by_tier[t] for t in business_tier_probs))
    max_value = float(base_value_by_tier["Enterprise"] * (1.0 + kappa * 1.0))

    pi_persona = personas.get("pi_persona", {})
    persona_rows = personas.get("personas", [])
    tau_lookup = {p.get("label"): float(p.get("tau_mean", 0.4)) for p in persona_rows}
    weighted_tau = 0.0
    for label, prob in pi_persona.items():
        weighted_tau += float(prob) * tau_lookup.get(label, 0.4)

    if weighted_tau <= 0:
        weighted_tau = 0.5

    churn_states = [
        ChurnState("S1", frustration=0.10, failed_streak=0, turn_count=5, tau=0.745, expected_lo=0.00, expected_hi=0.03),
        ChurnState("S2", frustration=0.30, failed_streak=0, turn_count=10, tau=0.607, expected_lo=0.03, expected_hi=0.08),
        ChurnState("S3", frustration=0.50, failed_streak=1, turn_count=15, tau=0.607, expected_lo=0.08, expected_hi=0.18),
        ChurnState("S4", frustration=0.70, failed_streak=2, turn_count=18, tau=0.332, expected_lo=0.18, expected_hi=0.40),
        ChurnState("S5", frustration=0.85, failed_streak=3, turn_count=20, tau=0.234, expected_lo=0.40, expected_hi=1.00),
        ChurnState("S6", frustration=0.95, failed_streak=4, turn_count=20, tau=0.174, expected_lo=0.60, expected_hi=1.00),
    ]

    churn_fit = calibrate_churn_coefficients(churn_states)
    coeffs = churn_fit["selected"]

    moderate_state = {"frustration": 0.55, "failed_streak": 1, "turn_count": 12, "tau": weighted_tau}
    p_churn_moderate = sigmoid(
        coeffs["c0"]
        + coeffs["cf"] * moderate_state["frustration"]
        + coeffs["cs"] * moderate_state["failed_streak"]
        + coeffs["ct"] * moderate_state["turn_count"]
        + coeffs["ctau"] * moderate_state["tau"]
    )

    # Fixed anchors from design: eta and terminal worst-case cap.
    eta = 5.0
    lambda_turn = 0.10
    t_max = 20
    v_max_for_cap = float(base_value_by_tier["Enterprise"] * (1.0 + kappa * 1.0))
    omega = float((eta - lambda_turn * t_max) / max(1e-9, v_max_for_cap))
    max_churn_cost = float(omega * v_max_for_cap)

    reward_params = {
        "eta_success": eta,
        "lambda_turn": lambda_turn,
        "omega": omega,
    }

    def p_churn(f: float, s: int, t: int, tau: float) -> float:
        z = coeffs["c0"] + coeffs["cf"] * f + coeffs["cs"] * s + coeffs["ct"] * t + coeffs["ctau"] * tau
        return sigmoid(z)

    def terminal_reward(
        *,
        tier: str,
        value_for_reward: float,
        p_churn_terminal: float,
        outcome: str,
    ) -> float:
        r = -reward_params["omega"] * p_churn_terminal * value_for_reward
        if outcome == "success":
            r += reward_params["eta_success"]
        elif outcome == "escalate":
            r -= escalation_cost_by_tier[tier]
        return float(r)

    trajectories = {
        "success_pro": {
            "tier": "Pro",
            "turns": 10,
            "outcome": "success",
            "p_churn_terminal": 0.0,
        },
        "escalate_pro": {
            "tier": "Pro",
            "turns": 15,
            "outcome": "escalate",
            "p_churn_terminal": 0.45,
        },
        "dropout_gold": {
            "tier": "Enterprise",
            "turns": 20,
            "outcome": "dropout",
            "p_churn_terminal": 1.0,
        },
        "escalate_gold": {
            "tier": "Enterprise",
            "turns": 10,
            "outcome": "escalate",
            "p_churn_terminal": 0.60,
        },
    }

    trajectory_results: dict[str, Any] = {}
    for name, cfg in trajectories.items():
        tier = str(cfg["tier"])
        turns = int(cfg["turns"])
        outcome = str(cfg["outcome"])
        p_term = float(cfg["p_churn_terminal"])

        if tier == "Enterprise":
            value_for_reward = v_max_for_cap
        else:
            value_for_reward = expected_value_by_tier[tier]

        per_turn_total = float(-reward_params["lambda_turn"] * turns)
        terminal = terminal_reward(
            tier=tier,
            value_for_reward=value_for_reward,
            p_churn_terminal=p_term,
            outcome=outcome,
        )
        raw_total = float(per_turn_total + terminal)
        total = raw_total
        trajectory_results[name] = {
            "tier": tier,
            "turns": turns,
            "p_churn_terminal": p_term,
            "value_for_reward": value_for_reward,
            "per_turn_total": per_turn_total,
            "terminal_reward": terminal,
            "raw_episode_total": raw_total,
            "episode_total": total,
            "required_clipping": bool(abs(raw_total - total) > 1e-12),
            "within_target_range_-5_to_5": bool(-5.0 <= total <= 5.0),
        }

    trajectory_checks = {
        "all_4_within_-5_to_5": bool(all(v["within_target_range_-5_to_5"] for v in trajectory_results.values())),
        "no_clipping_required_all": bool(all(not v["required_clipping"] for v in trajectory_results.values())),
        "success_pro_gt_3": bool(trajectory_results["success_pro"]["episode_total"] > 3.0),
        "escalate_pro_lt_-1": bool(trajectory_results["escalate_pro"]["episode_total"] < -1.0),
        "dropout_gold_lt_-3": bool(trajectory_results["dropout_gold"]["episode_total"] < -3.0),
        "escalate_gold_lt_escalate_pro": bool(
            trajectory_results["escalate_gold"]["episode_total"]
            < trajectory_results["escalate_pro"]["episode_total"]
        ),
    }

    warnings: list[str] = []
    if not churn_fit["defaults_valid"]:
        warnings.append("Default churn coefficients failed target bands; calibrated coefficients were used.")
    if not churn_fit["selected_valid"]:
        warnings.append("Calibrated churn coefficients are only approximate; one or more state checks remains outside band.")
    if max_churn_cost > eta:
        warnings.append("Max churn cost constraint violated: omega * V_max >= eta.")
    warnings.append("Per-turn churn accumulation replaced with terminal-only churn cost to prevent reward-scale overflow.")
    if any(v["required_clipping"] for v in trajectory_results.values()):
        warnings.append("One or more trajectories required clipping, which violates the Phase 6 no-clipping target.")
    for name, res in trajectory_results.items():
        if not res["within_target_range_-5_to_5"]:
            warnings.append(f"Trajectory {name} total reward out of [-5, 5].")
    for check_name, check_ok in trajectory_checks.items():
        if not check_ok:
            warnings.append(f"Trajectory check failed: {check_name}")

    member_level_config: dict[str, Any] = {}
    member_level_value_weight = {
        "guest": 0.05,
        "bronze": 0.425,
        "silver": 0.75,
        "gold": 1.0,
    }
    for lvl, p in tier_freqs.items():
        if lvl in tier_mapping:
            member_level_config[lvl] = {
                "business_tier": tier_mapping[lvl],
                "init_probability": float(p),
                "value_weight": member_level_value_weight[lvl],
                "included_in_calibration": True,
            }
        elif lvl in excluded_member_levels:
            member_level_config[lvl] = {
                "business_tier": None,
                "init_probability": float(p),
                "value_weight": None,
                "included_in_calibration": False,
                "note": excluded_member_levels[lvl],
            }

    tier_payload = {
        "tier_mapping": tier_mapping,
        "excluded_member_levels": excluded_member_levels,
        "member_level_config": member_level_config,
        "member_level_init_frequencies": tier_freqs,
        "business_tier_probabilities": business_tier_probs,
        "base_value_by_tier": base_value_by_tier,
        "representative_value_weight": representative_value_weight,
        "kappa": kappa,
        "expected_value_by_tier": expected_value_by_tier,
        "escalation_cost_by_tier": escalation_cost_by_tier,
        "notes": {
            "source": "Phase 2 tier frequencies with deterministic ABCD-to-business mapping",
            "normalization": "business tier probabilities renormalized to sum to 1",
            "exclusions": "platinum and vip are excluded from calibration due to 0% frequency",
        },
    }

    churn_payload = {
        "formula": "sigmoid(c0 + cf*f + cs*failed_streak + ct*turn_count + ctau*tau)",
        "coefficients": coeffs,
        "default_coefficients": churn_fit["defaults"],
        "defaults_valid": churn_fit["defaults_valid"],
        "state_validation": churn_fit["state_checks"],
    }

    reward_payload = {
        "reward_formula": {
            "per_turn": "R_t = -lambda_turn",
            "terminal": "R_terminal = eta*1_success - C_e(tier)*1_escalate - omega*P_churn_terminal*V",
            "P_churn_terminal_by_outcome": {
                "success": 0.0,
                "escalation": "computed P_churn at time of escalation",
                "dropout": 1.0,
                "timeout": "computed P_churn at turn T_max",
            },
        },
        "parameters": reward_params,
        "calibration": {
            "moderate_state": moderate_state,
            "p_churn_moderate": float(p_churn_moderate),
            "avg_value_reference": avg_value,
            "omega_selected": float(omega),
            "max_churn_cost": max_churn_cost,
            "max_churn_cost_constraint": "max_churn_cost < eta",
            "eta": eta,
            "T_max": t_max,
            "V_max_for_cap": v_max_for_cap,
            "V_max_from_value_function": max_value,
            "lambda_turn_selected": reward_params["lambda_turn"],
            "omega_formula": "omega = (eta - lambda_turn*T_max) / V_max",
        },
        "trajectory_sanity": trajectory_results,
        "trajectory_checks": trajectory_checks,
        "trajectory_scale_note": "No clipping is applied; episode_total equals raw_episode_total for all trajectories.",
        "diagnostics": {
            "transition_dropout_empirical_rate_phase3": transition.get("dropout", {}).get("empirical_dropout_rate"),
            "extract1_resolution_rate": float(extract1["resolution_flag"].mean()),
            "extract1_escalation_rate": float(extract1["escalation_flag"].mean()),
            "extract3_mean_subflow_resolution_rate": float(extract3["resolution_rate"].mean()),
        },
        "warnings": warnings,
    }

    (phase6_dir / "tier_config.json").write_text(json.dumps(tier_payload, indent=2), encoding="utf-8")
    (phase6_dir / "churn_validation.json").write_text(json.dumps(churn_payload, indent=2), encoding="utf-8")
    (phase6_dir / "reward_model.json").write_text(json.dumps(reward_payload, indent=2), encoding="utf-8")

    print("Saved:")
    print(f"- {phase6_dir / 'tier_config.json'}")
    print(f"- {phase6_dir / 'churn_validation.json'}")
    print(f"- {phase6_dir / 'reward_model.json'}")
    print("\nKey outputs:")
    print(f"omega={omega:.6f}")
    print(f"max_churn_cost={max_churn_cost:.6f}")
    print("state_validation:")
    for row in churn_fit["state_checks"]:
        print(f"  {row['state']}: p={row['p_churn']:.4f} in {row['expected_range']} -> {row['within_expected_range']}")
    print("trajectory_totals:")
    for name, result in trajectory_results.items():
        print(f"  {name}: {result['episode_total']:.4f} (raw={result['raw_episode_total']:.4f})")
    print("trajectory_checks:")
    for k, v in trajectory_checks.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
