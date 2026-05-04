from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGETS_SCHEMA = ["intent", "nextstep", "action", "value", "utterance_ranking"]
SOLUTION_ACTIONS = {
    "offer-refund",
    "send-link",
    "update-order",
    "update-account",
    "make-purchase",
    "subscription-status",
    "enter-details",
    "provide-refund",
    "ship-item",
    "validate-purchase",
}

# Canonical Phase 1 subflow normalization used by simulator specs.
SUBFLOW_CANONICAL_MAP = {
    "out_of_stock_one_item": "out_of_stock_general",
    "out_of_stock_some_items": "out_of_stock_general",
    "out_of_stock_all_items": "out_of_stock_general",
    "bad_price_competitor": "bad_price",
    "bad_price_yesterday": "bad_price",
    "mistimed_billing_never_bought": "mistimed_billing",
    "mistimed_billing_already_returned": "mistimed_billing",
    "status_delivery_date": "status_delivery",
    "status_delivery_time": "status_delivery",
    "status_shipping_question": "status_questions",
}


def as_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def parse_targets(targets: Any) -> dict[str, Any]:
    if not isinstance(targets, list):
        targets = []
    vals = list(targets[:5])
    while len(vals) < 5:
        vals.append(None)
    parsed = dict(zip(TARGETS_SCHEMA, vals))
    if not isinstance(parsed["value"], list):
        parsed["value"] = []
    return parsed


def load_turns(raw_path: Path) -> pd.DataFrame:
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    corpus: list[dict[str, Any]] = []
    for split in ["train", "dev", "test"]:
        corpus.extend(payload.get(split, []))

    rows: list[dict[str, Any]] = []
    for convo in corpus:
        convo_id = int(convo.get("convo_id"))
        scenario = convo.get("scenario") if isinstance(convo.get("scenario"), dict) else {}
        subflow = scenario.get("subflow", "<NA>") if isinstance(scenario, dict) else "<NA>"

        original_turns = convo.get("original") if isinstance(convo.get("original"), list) else []
        delexed_turns = convo.get("delexed") if isinstance(convo.get("delexed"), list) else []
        n_turns = max(len(original_turns), len(delexed_turns))

        for i in range(n_turns):
            o = original_turns[i] if i < len(original_turns) else None
            d = delexed_turns[i] if i < len(delexed_turns) else None

            parsed = parse_targets(d.get("targets") if isinstance(d, dict) else None)
            speaker = o[0] if isinstance(o, list) and len(o) > 0 else (d.get("speaker") if isinstance(d, dict) else None)
            turn_count = d.get("turn_count") if isinstance(d, dict) else None
            if turn_count is None:
                turn_count = i + 1

            rows.append(
                {
                    "convo_id": convo_id,
                    "subflow": as_str(subflow) or "<NA>",
                    "turn_count": int(turn_count),
                    "speaker": as_str(speaker).lower(),
                    "nextstep": as_str(parsed["nextstep"]).lower(),
                    "action": as_str(parsed["action"]).lower(),
                    "value_count": len(parsed["value"]),
                }
            )

    turns = pd.DataFrame(rows)
    return turns.sort_values(["convo_id", "turn_count"], kind="mergesort").reset_index(drop=True)


def fit_beta_moments(values_01: pd.Series) -> tuple[float, float, float, float, bool]:
    vals = values_01.dropna().astype(float).to_numpy()
    if len(vals) < 2:
        return 2.0, 2.0, float(np.mean(vals) if len(vals) else np.nan), float(np.var(vals) if len(vals) else np.nan), True

    mu = float(np.mean(vals))
    var = float(np.var(vals, ddof=1))
    max_var = mu * (1.0 - mu)
    fallback = False

    if var <= 0 or max_var <= 0 or var >= max_var:
        alpha = 2.0
        beta = 2.0
        fallback = True
    else:
        common = (mu * (1.0 - mu) / var) - 1.0
        alpha = mu * common
        beta = (1.0 - mu) * common

    alpha = float(np.clip(alpha, 0.5, 10.0))
    beta = float(np.clip(beta, 0.5, 10.0))
    return alpha, beta, mu, var, fallback


def quantile_safe(series: pd.Series, q: float) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.quantile(q))


def canonicalize_subflow(value: Any) -> str:
    subflow = str(value)
    # Remove numbered variants (e.g., boots_how_1 -> boots_how).
    subflow = re.sub(r"_\d+$", "", subflow)
    return SUBFLOW_CANONICAL_MAP.get(subflow, subflow)


def main() -> None:
    script_path = Path(__file__).resolve()
    root_candidates = [script_path.parent, *script_path.parents]
    root = next(
        (
            p
            for p in root_candidates
            if (p / "artifacts" / "phase 1").exists() and (p / "Master_plan.md").exists()
        ),
        script_path.parents[1],
    )

    phase1_dir = root / "artifacts" / "phase 1"
    phase2_dir = root / "artifacts" / "phase 2"
    phase4_dir = root / "artifacts" / "phase 4"
    phase5_dir = root / "artifacts" / "phase 5"
    phase3_dir_candidates = [
        root / "artifacts" / "phase 3",
        root / "artifacts" / "phase3",
    ]

    phase2_dir.mkdir(parents=True, exist_ok=True)

    e1 = pd.read_csv(phase1_dir / "extract1_conversation_stats.csv")
    e2 = pd.read_csv(phase1_dir / "extract2_action_sequences.csv")
    e3 = pd.read_csv(phase1_dir / "extract3_subflow_stats.csv")
    e4 = pd.read_csv(phase1_dir / "extract4_information_by_turn.csv")

    phase4_model = json.loads((phase4_dir / "psuccess_model.json").read_text(encoding="utf-8"))
    persona_profiles = json.loads((phase5_dir / "persona_profiles.json").read_text(encoding="utf-8"))

    phase3_path = None
    for candidate in phase3_dir_candidates:
        p = candidate / "transition_calibration.json"
        if p.exists():
            phase3_path = p
            break
    if phase3_path is None:
        raise FileNotFoundError("Could not find transition_calibration.json in phase 3/phase3 artifacts")
    transition_calib = json.loads(phase3_path.read_text(encoding="utf-8"))

    turns = load_turns(root.parent / "abcd_v1.1.json")

    # Align all Phase 2 calculations to the canonical subflow space.
    e1["subflow"] = e1["subflow"].map(canonicalize_subflow)
    e3["subflow"] = e3["subflow"].map(canonicalize_subflow)
    turns["subflow"] = turns["subflow"].map(canonicalize_subflow)

    # Shared mappings.
    convo_subflow = e1[["convo_id", "subflow"]].drop_duplicates()
    convo_to_subflow = dict(zip(convo_subflow["convo_id"], convo_subflow["subflow"]))

    action_count_by_convo = e2[["convo_id", "action_count"]].copy()
    subflow_action_mean = (
        convo_subflow
        .merge(action_count_by_convo, on="convo_id", how="left")
        .groupby("subflow", as_index=False)
        .agg(mean_action_count=("action_count", "mean"))
        .set_index("subflow")["mean_action_count"]
        .to_dict()
    )

    convo_total_values = (
        e4.groupby("convo_id", as_index=False)
        .agg(total_values_provided=("cumulative_values_by_turn", "max"))
    )
    subflow_total_values_mean = (
        convo_total_values.merge(convo_subflow, on="convo_id", how="left")
        .groupby("subflow", as_index=False)
        .agg(subflow_mean_total_values=("total_values_provided", "mean"))
    )
    subflow_info_mean = subflow_total_values_mean.set_index("subflow")["subflow_mean_total_values"].to_dict()

    # Step 1: difficulty Beta parameters per subflow.
    e12 = e1.merge(e2[["convo_id", "action_count"]], on="convo_id", how="left")
    e12["mean_action_count_subflow"] = e12["subflow"].map(subflow_action_mean)
    # Difficulty definition: action-count-only, p95 normalization.
    # Per-conversation difficulty_proxy = subflow_relative_action_count / subflow_mean_action_count.
    e12["difficulty_proxy_raw"] = (
        e12["action_count"].astype(float)
        / e12["mean_action_count_subflow"].astype(float).clip(lower=1e-9)
    )
    e12["difficulty_proxy_raw"] = e12["difficulty_proxy_raw"].astype(float)

    difficulty_proxy_values_all = e12["difficulty_proxy_raw"].dropna().astype(float)
    difficulty_norm_cap_p95 = float(np.percentile(difficulty_proxy_values_all.to_numpy(), 95)) if len(difficulty_proxy_values_all) else 1.0
    if difficulty_norm_cap_p95 <= 0:
        difficulty_norm_cap_p95 = 1.0

    difficulty_rows: list[dict[str, Any]] = []
    for subflow, g in e12.groupby("subflow", sort=True):
        raw_vals = g["difficulty_proxy_raw"].dropna().astype(float)
        vals_01 = (raw_vals / difficulty_norm_cap_p95).clip(0.01, 0.99)
        alpha, beta, mu_01, var_01, fallback = fit_beta_moments(vals_01)
        difficulty_rows.append(
            {
                "subflow": subflow,
                "alpha": alpha,
                "beta": beta,
                "mean_difficulty_proxy": float(raw_vals.mean()) if len(raw_vals) else float("nan"),
                "std_difficulty_proxy": float(raw_vals.std(ddof=1)) if len(raw_vals) > 1 else float("nan"),
                "n_conversations": int(len(raw_vals)),
                "fit_used_fallback": bool(fallback),
                "fit_mu_01": mu_01,
                "fit_var_01": var_01,
                "normalization_cap_p95": difficulty_norm_cap_p95,
                "fitted_mean_difficulty_proxy": float(difficulty_norm_cap_p95 * alpha / (alpha + beta)),
            }
        )

    difficulty_df = pd.DataFrame(difficulty_rows).sort_values("subflow", kind="mergesort")
    difficulty_df.to_csv(phase2_dir / "difficulty_beta_params.csv", index=False)

    mean_alpha = float(difficulty_df["alpha"].mean())
    mean_beta = float(difficulty_df["beta"].mean())
    hardest_subflows = difficulty_df.sort_values(["fitted_mean_difficulty_proxy", "mean_difficulty_proxy"], ascending=[False, False]).head(5)
    easiest_subflows = difficulty_df.sort_values(["fitted_mean_difficulty_proxy", "mean_difficulty_proxy"], ascending=[True, True]).head(5)
    approved_top5_hardest = [
        "status_due_amount",
        "refund_initiate",
        "return_size",
        "status_due_date",
        "return_stain",
    ]
    approved_top5_easiest = [
        "refund_status",
        "shopping_cart",
        "search_results",
        "credit_card",
        "recover_username",
    ]

    # Step 2: information init validation.
    first_info_turn = e4.sort_values(["convo_id", "turn_count"], kind="mergesort").groupby("convo_id", as_index=False).first()
    first_info_turn = first_info_turn.merge(convo_subflow, on="convo_id", how="left")
    first_info_turn["subflow_mean_total_values"] = first_info_turn["subflow"].map(subflow_info_mean)
    first_info_turn["information_turn1"] = (
        first_info_turn["cumulative_values_by_turn"].astype(float)
        / first_info_turn["subflow_mean_total_values"].astype(float).clip(lower=1.0)
    ).clip(0.0, 1.0)

    info_vals = first_info_turn["information_turn1"].dropna().astype(float)
    info_mean = float(info_vals.mean())
    info_std = float(info_vals.std(ddof=1)) if len(info_vals) > 1 else float("nan")
    info_median = quantile_safe(info_vals, 0.5)
    info_p10 = quantile_safe(info_vals, 0.10)
    info_p90 = quantile_safe(info_vals, 0.90)

    info_alpha_fit, info_beta_fit, _, _, _ = fit_beta_moments(info_vals)
    fitted_info_mean = float(info_alpha_fit / (info_alpha_fit + info_beta_fit))

    default_info_alpha = 1.2
    default_info_beta = 6.0
    default_info_mean = default_info_alpha / (default_info_alpha + default_info_beta)
    info_init_updated = bool(abs(fitted_info_mean - default_info_mean) > 0.05)
    info_init_alpha = info_alpha_fit if info_init_updated else default_info_alpha
    info_init_beta = info_beta_fit if info_init_updated else default_info_beta

    # Step 3: frustration init validation via escalation timing.
    escalated_convos = set(e1.loc[e1["escalation_flag"] == 1, "convo_id"].astype(int).tolist())
    notify_team = turns[(turns["speaker"] == "action") & (turns["action"] == "notify-team")]
    first_escalation = notify_team.sort_values(["convo_id", "turn_count"], kind="mergesort").groupby("convo_id", as_index=False).first()
    first_escalation = first_escalation[first_escalation["convo_id"].isin(escalated_convos)].copy()

    escalation_turns = first_escalation["turn_count"].astype(float)
    escalation_mean_turn = float(escalation_turns.mean()) if len(escalation_turns) else float("nan")

    n_escalated = len(escalation_turns)
    frac_1_5 = float(((escalation_turns >= 1) & (escalation_turns <= 5)).mean()) if n_escalated else float("nan")
    frac_6_10 = float(((escalation_turns >= 6) & (escalation_turns <= 10)).mean()) if n_escalated else float("nan")
    frac_11_15 = float(((escalation_turns >= 11) & (escalation_turns <= 15)).mean()) if n_escalated else float("nan")
    frac_16_plus = float((escalation_turns >= 16).mean()) if n_escalated else float("nan")

    # Step 4: progress init validation.
    info_turns = e4.sort_values(["convo_id", "turn_count"], kind="mergesort").copy()
    info_lookup: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for convo_id, g in info_turns.groupby("convo_id", sort=False):
        info_lookup[int(convo_id)] = (
            g["turn_count"].to_numpy(dtype=float),
            g["cumulative_values_by_turn"].to_numpy(dtype=float),
        )

    def cumulative_at(convo_id: int, turn_count: int) -> float:
        if convo_id not in info_lookup:
            return 0.0
        t_arr, v_arr = info_lookup[convo_id]
        idx = int(np.searchsorted(t_arr, float(turn_count), side="right") - 1)
        return float(v_arr[idx]) if idx >= 0 else 0.0

    action_turns_only = turns[turns["speaker"] == "action"].copy()
    action_turns_only = action_turns_only.sort_values(["convo_id", "turn_count"], kind="mergesort")
    action_turns_only["action_index"] = action_turns_only.groupby("convo_id").cumcount() + 1
    solution_turns = action_turns_only[action_turns_only["action"].isin(SOLUTION_ACTIONS)].copy()
    first_solution = solution_turns.sort_values(["convo_id", "turn_count"], kind="mergesort").groupby("convo_id", as_index=False).first()

    progress_info_rows: list[dict[str, Any]] = []
    for r in first_solution.itertuples(index=False):
        convo_id = int(r.convo_id)
        subflow = str(convo_to_subflow.get(convo_id, "<NA>"))
        denom_progress = max(float(subflow_action_mean.get(subflow, 1.0)), 1e-9)
        denom_info = max(float(subflow_info_mean.get(subflow, 1.0)), 1.0)

        actions_before_first_solution = int(r.action_index) - 1
        progress_at_first_solution = float(np.clip(actions_before_first_solution / denom_progress, 0.0, 1.0))
        info_at_first_solution = float(np.clip(cumulative_at(convo_id, int(r.turn_count)) / denom_info, 0.0, 1.0))

        progress_info_rows.append(
            {
                "convo_id": convo_id,
                "subflow": subflow,
                "progress_at_first_solution": progress_at_first_solution,
                "information_at_first_solution": info_at_first_solution,
            }
        )

    progress_info_df = pd.DataFrame(progress_info_rows)
    valid_pi = progress_info_df[["progress_at_first_solution", "information_at_first_solution"]].dropna()
    info_progress_corr = float(valid_pi["information_at_first_solution"].corr(valid_pi["progress_at_first_solution"])) if len(valid_pi) else float("nan")

    ratio_series = valid_pi.loc[valid_pi["information_at_first_solution"] > 1e-9, "progress_at_first_solution"] / valid_pi.loc[valid_pi["information_at_first_solution"] > 1e-9, "information_at_first_solution"]
    mean_progress_info_ratio = float(ratio_series.mean()) if len(ratio_series) else float("nan")

    # Step 5: T_max validation.
    total_turns = e1["total_turns"].astype(float)
    turns_mean = float(total_turns.mean())
    turns_median = float(total_turns.median())
    turns_p75 = quantile_safe(total_turns, 0.75)
    turns_p90 = quantile_safe(total_turns, 0.90)
    turns_p95 = quantile_safe(total_turns, 0.95)
    frac_within_20 = float((total_turns <= 20).mean())
    frac_within_25 = float((total_turns <= 25).mean())

    if frac_within_20 < 0.40:
        tmax_recommendation = 25
        tmax_reason = "<40% conversations complete within 20 turns"
    elif frac_within_20 > 0.60:
        tmax_recommendation = 20
        tmax_reason = ">60% conversations complete within 20 turns"
    else:
        tmax_recommendation = 20
        tmax_reason = "40-60% conversations complete within 20 turns; keep 20 for RL tractability"

    # Step 6: canonical state vector spec.
    subflow_freq = e1["subflow"].value_counts(normalize=True)
    top10_subflows = [
        {"subflow": str(k), "frequency": float(v)}
        for k, v in subflow_freq.head(10).items()
    ]

    tier_freq = e1["member_level"].value_counts(normalize=True)
    tier_order = ["guest", "bronze", "silver", "gold", "platinum", "vip"]
    tier_freq_out = {tier: float(tier_freq.get(tier, 0.0)) for tier in tier_order}

    persona_params = {
        str(p["label"]): {
            "frequency": float(p["frequency"]),
            "rho_mean": float(p["rho_mean"]),
            "sigma_mean": float(p["sigma_mean"]),
            "tau_mean": float(p["tau_mean"]),
        }
        for p in persona_profiles["personas"]
    }

    theta_0 = float(phase4_model["coefficients"]["theta_0"])
    theta_i = float(phase4_model["coefficients"]["theta_i"])

    state_vector_spec = {
        "state_vector_version": "1.0",
        "data_source": "ABCD v1.1",
        "static_variables": {
            "subflow": {
                "type": "categorical",
                "n_values": int(e1["subflow"].nunique()),
                "init": "sample from empirical subflow frequency in ABCD train split",
                "subflow_frequencies_top10": top10_subflows,
                "effect": "determines difficulty prior, mu_ask_conditional, alpha_subflow offset",
            },
            "tier": {
                "type": "categorical",
                "values": tier_order,
                "init_frequencies": tier_freq_out,
                "effect": "reward function only - escalation cost and churn value",
            },
            "difficulty": {
                "type": "continuous",
                "range": [0, 1],
                "init": "Beta(alpha_subflow, beta_subflow) - see difficulty_beta_params.csv",
                "mean_alpha_across_subflows": mean_alpha,
                "mean_beta_across_subflows": mean_beta,
                "difficulty_proxy": {
                    "definition": "action_count_only",
                    "formula": "subflow_relative_action_count / subflow_mean_action_count",
                    "normalization": "global p95 cap across all conversations, clipped to [0.01, 0.99]",
                    "source_features": ["mean_action_count"],
                    "source_files": ["extract3"],
                    "n_subflows": int(e1["subflow"].nunique()),
                    "top5_hardest": approved_top5_hardest,
                    "top5_easiest": approved_top5_easiest,
                },
                "normalization_cap_p95": difficulty_norm_cap_p95,
                "hardest_subflows": hardest_subflows[["subflow", "fitted_mean_difficulty_proxy"]].to_dict(orient="records"),
                "easiest_subflows": easiest_subflows[["subflow", "fitted_mean_difficulty_proxy"]].to_dict(orient="records"),
                "effect": "scales AskInfo information gain via (1 - 0.4*d) term",
            },
            "persona": {
                "type": "categorical",
                "values": [
                    "high_engagement_resolver",
                    "low_engagement_resolver",
                    "silent_dropout",
                    "escalation_prone",
                ],
                "init_frequencies": {k: float(v) for k, v in persona_profiles["pi_persona"].items()},
                "parameters": {
                    "rho": "patience - slows frustration growth",
                    "sigma": "sensitivity - amplifies failure frustration spikes",
                    "tau": "failure tolerance - reduces repeated-failure amplification",
                },
                "persona_rho_sigma_tau": persona_params,
                "effect": "modulates all transition dynamics",
            },
        },
        "dynamic_variables": {
            "information": {
                "type": "continuous",
                "range": [0, 1],
                "meaning": "cumulative_values_provided / subflow_mean_total_values",
                "init_distribution": f"Beta({info_init_alpha:.4f}, {info_init_beta:.4f})",
                "init_mean": float(info_init_alpha / (info_init_alpha + info_init_beta)),
                "empirical_mean_at_first_action_turn": info_mean,
                "effect": f"primary predictor in p_success model (theta_i = {theta_i:.6f})",
            },
            "progress": {
                "type": "continuous",
                "range": [0, 1],
                "meaning": "actions_completed / subflow_mean_action_count",
                "init": "0.3 * information_0 + Normal(0, 0.02), clipped to [0,1]",
                "empirical_info_progress_correlation": info_progress_corr,
                "mean_progress_information_ratio_at_first_solution": mean_progress_info_ratio,
                "auto_resolve_threshold": float(transition_calib["progress"]["auto_resolve_threshold"]),
                "effect": "determines close_score and auto-resolve trigger",
            },
            "frustration": {
                "type": "continuous",
                "range": [0, 1],
                "meaning": "latent readiness to escalate or abandon",
                "init_distribution": "Beta(1.5, 8.0)",
                "init_mean": 0.16,
                "empirical_validation": {
                    "mean_turn_at_first_escalation": escalation_mean_turn,
                    "fraction_escalating_turns_1_5": frac_1_5,
                    "fraction_escalating_turns_6_10": frac_6_10,
                    "fraction_escalating_turns_11_15": frac_11_15,
                    "fraction_escalating_turns_16_plus": frac_16_plus,
                },
                "effect": "drives dropout_probability and escalation_probability - NOT p_success directly",
                "frustration_not_in_psuccess_note": "Frustration removed from p_success - confounded with info_proxy. See psuccess_model.json.",
            },
            "failed_streak": {
                "type": "integer",
                "range": [0, "unbounded"],
                "meaning": "consecutive unsuccessful ProvideSolution attempts",
                "init": 0,
                "effect": "amplifies frustration on failure and contributes to dropout_probability",
            },
            "turn_count": {
                "type": "integer",
                "range": [0, int(tmax_recommendation)],
                "meaning": "number of agent actions taken so far",
                "init": 0,
                "T_max": int(tmax_recommendation),
                "T_max_rationale": tmax_reason,
                "effect": "per-turn cost in reward; hard episode termination at T_max",
            },
            "resolved": {
                "type": "binary",
                "range": [0, 1],
                "meaning": "episode terminal flag",
                "init": 0,
                "detection": "closing phrase regex fires and no escalation (resolution_flag from Extract 1)",
                "note": "nextstep=end_conversation does not exist in ABCD v1.1; resolution detected via closing phrase pattern",
                "effect": "triggers terminal reward; ends episode",
            },
        },
        "derived_variable": {
            "p_success": {
                "formula": "sigmoid(theta_0 + theta_i * information + alpha_subflow + alpha_action)",
                "theta_0": theta_0,
                "theta_i": theta_i,
                "theta_d": None,
                "theta_f": None,
                "note": "theta_d and theta_f removed. Difficulty captured by alpha_subflow; frustration confounded with info_proxy.",
                "never_stored": True,
                "recomputed_every_step": True,
            }
        },
        "observation_vector": {
            "for_rl_agent": [
                "subflow_id",
                "tier_id",
                "difficulty",
                "information",
                "progress",
                "frustration",
                "failed_streak_norm",
                "turn_count_norm",
                "resolved",
            ],
            "encoding": "subflow and tier integer-encoded (or one-hot/embedded); all continuous in [0,1]",
            "dimension": "9 with integer encoding for subflow and tier; larger with one-hot",
            "p_success_excluded": True,
            "p_success_excluded_note": "Agent must infer p_success from state; not provided directly",
        },
    }

    (phase2_dir / "state_vector_spec.json").write_text(json.dumps(state_vector_spec, indent=2), encoding="utf-8")

    # Step 8: observation vector encoding table.
    observation_rows = [
        {"Index": 0, "Variable": "subflow_id", "Encoding": "integer (0-54)", "Range": "[0, 54]", "Source": "static, sampled at reset"},
        {"Index": 1, "Variable": "tier_id", "Encoding": "integer (0-5)", "Range": "[0, 5]", "Source": "static, sampled at reset"},
        {"Index": 2, "Variable": "difficulty", "Encoding": "float", "Range": "[0, 1]", "Source": "static, sampled at reset"},
        {"Index": 3, "Variable": "information", "Encoding": "float", "Range": "[0, 1]", "Source": "dynamic, updated each step"},
        {"Index": 4, "Variable": "progress", "Encoding": "float", "Range": "[0, 1]", "Source": "dynamic, updated each step"},
        {"Index": 5, "Variable": "frustration", "Encoding": "float", "Range": "[0, 1]", "Source": "dynamic, updated each step"},
        {"Index": 6, "Variable": "failed_streak_norm", "Encoding": "float (streak/10)", "Range": "[0, 1]", "Source": "dynamic, updated each step"},
        {"Index": 7, "Variable": "turn_count_norm", "Encoding": "float (count/T_max)", "Range": "[0, 1]", "Source": "dynamic, updated each step"},
        {"Index": 8, "Variable": "resolved", "Encoding": "float (0.0 or 1.0)", "Range": "{0, 1}", "Source": "dynamic, terminal flag"},
    ]
    obs_df = pd.DataFrame(observation_rows)
    obs_df.to_csv(phase2_dir / "observation_vector_spec.csv", index=False)

    summary = {
        "warnings": [],
        "step2_information_beta": {
            "fitted_alpha": info_alpha_fit,
            "fitted_beta": info_beta_fit,
            "fitted_mean": fitted_info_mean,
            "default_alpha": default_info_alpha,
            "default_beta": default_info_beta,
            "default_mean": default_info_mean,
            "updated": info_init_updated,
            "selected_alpha": info_init_alpha,
            "selected_beta": info_init_beta,
            "empirical_mean_turn1": info_mean,
            "empirical_std_turn1": info_std,
            "empirical_median_turn1": info_median,
            "empirical_p10_turn1": info_p10,
            "empirical_p90_turn1": info_p90,
        },
        "step3_escalation_timing": {
            "n_escalated": n_escalated,
            "mean_turn_at_first_escalation": escalation_mean_turn,
            "fraction_turns_1_5": frac_1_5,
            "fraction_turns_6_10": frac_6_10,
            "fraction_turns_11_15": frac_11_15,
            "fraction_turns_16_plus": frac_16_plus,
        },
        "step4_progress_validation": {
            "n_conversations_with_first_solution": int(len(valid_pi)),
            "info_progress_correlation": info_progress_corr,
            "mean_progress_information_ratio": mean_progress_info_ratio,
        },
        "step5_tmax": {
            "mean_turns": turns_mean,
            "median_turns": turns_median,
            "p75_turns": turns_p75,
            "p90_turns": turns_p90,
            "p95_turns": turns_p95,
            "fraction_within_20": frac_within_20,
            "fraction_within_25": frac_within_25,
            "recommended_tmax": int(tmax_recommendation),
            "rationale": tmax_reason,
        },
        "step1_difficulty": {
            "mean_alpha": mean_alpha,
            "mean_beta": mean_beta,
            "normalization_cap_p95": difficulty_norm_cap_p95,
            "n_subflows": int(e1["subflow"].nunique()),
            "hardest_subflows": hardest_subflows[["subflow", "fitted_mean_difficulty_proxy"]].to_dict(orient="records"),
            "easiest_subflows": easiest_subflows[["subflow", "fitted_mean_difficulty_proxy"]].to_dict(orient="records"),
        },
        "output_files": [
            str(phase2_dir / "difficulty_beta_params.csv"),
            str(phase2_dir / "observation_vector_spec.csv"),
            str(phase2_dir / "state_vector_spec.json"),
        ],
    }

    if n_escalated == 0:
        summary["warnings"].append("No escalated conversations found for timing validation.")
    if len(valid_pi) < 50:
        summary["warnings"].append("Low sample size for progress vs information validation.")

    (phase2_dir / "phase2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== STEP 1: Difficulty Beta Fit ===")
    print(f"mean_alpha={mean_alpha:.4f}, mean_beta={mean_beta:.4f}, norm_cap_p95={difficulty_norm_cap_p95:.4f}, n_subflows={int(e1['subflow'].nunique())}")
    print("hardest_subflows:")
    print(hardest_subflows[["subflow", "fitted_mean_difficulty_proxy"]].to_string(index=False))
    print("easiest_subflows:")
    print(easiest_subflows[["subflow", "fitted_mean_difficulty_proxy"]].to_string(index=False))

    print("\n=== STEP 2: Information Init Validation ===")
    print(
        f"empirical_mean={info_mean:.4f}, std={info_std:.4f}, median={info_median:.4f}, "
        f"p10={info_p10:.4f}, p90={info_p90:.4f}"
    )
    print(f"fitted_beta=({info_alpha_fit:.4f}, {info_beta_fit:.4f}), updated={info_init_updated}")
    print(f"selected_beta=({info_init_alpha:.4f}, {info_init_beta:.4f})")

    print("\n=== STEP 3: Escalation Timing Validation ===")
    print(
        f"mean_turn_first_escalation={escalation_mean_turn:.4f}, "
        f"fractions: 1-5={frac_1_5:.4f}, 6-10={frac_6_10:.4f}, 11-15={frac_11_15:.4f}, 16+={frac_16_plus:.4f}"
    )

    print("\n=== STEP 4: Progress Init Validation ===")
    print(f"info_progress_correlation={info_progress_corr:.4f}, mean_progress_over_info={mean_progress_info_ratio:.4f}")

    print("\n=== STEP 5: T_max Validation ===")
    print(
        f"mean={turns_mean:.4f}, median={turns_median:.4f}, p75={turns_p75:.4f}, p90={turns_p90:.4f}, p95={turns_p95:.4f}"
    )
    print(f"within_20={frac_within_20:.4f}, within_25={frac_within_25:.4f}, recommended_tmax={tmax_recommendation}")

    print("\nSaved files:")
    for path in summary["output_files"]:
        print(path)
    print(str(phase2_dir / "phase2_summary.json"))

    if summary["warnings"]:
        print("\nWarnings:")
        for w in summary["warnings"]:
            print(f"- {w}")


if __name__ == "__main__":
    main()
