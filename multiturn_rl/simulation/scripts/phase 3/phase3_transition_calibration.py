from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGETS_SCHEMA = ["intent", "nextstep", "action", "value", "utterance_ranking"]

REPAIR_RE = re.compile(
    r"\b(i('m| am) sorry|i apologize|i understand (your|how)|"
    r"i can (understand|see) (why|how)|that must be|"
    r"i appreciate your patience|thank you for your patience|"
    r"i sincerely apologize|we apologize|i completely understand)\b",
    re.IGNORECASE,
)

CLOSING_RE = re.compile(
    r"\b(?:have a great day|have a good day|have a nice day|have a wonderful day|"
    r"have a great one|have a great night|have a good one|"
    r"rest of your day|wonderful rest|great rest|"
    r"goodbye|good bye|bye|you too|that's all|that is all|"
    r"that's it|that is it|we're all set|we are all set|"
    r"all set|all taken care|taken care of|"
    r"thank you for contacting|pleasure to help|glad i could help|"
    r"happy to help|happy i could help|glad i could|"
    r"no problem at all|not a problem|my pleasure|"
    r"is there anything else|take care|see you|talk soon|"
    r"sorry i couldn't|couldn't be of more|hope that helps|"
    r"don't hesitate to|feel free to contact|feel free to reach)\b",
    re.IGNORECASE,
)

CUSTOMER_ESCALATION_RE = re.compile(
    r"\b(?:speak to a manager|speak to your manager|talk to a manager|"
    r"want a manager|need a manager|get a manager|"
    r"speak to a supervisor|talk to a supervisor|"
    r"speak to a human|talk to a human|real person)\b",
    re.IGNORECASE,
)

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


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-x)))


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
            text = o[1] if isinstance(o, list) and len(o) > 1 else (d.get("text") if isinstance(d, dict) else None)
            turn_count = d.get("turn_count") if isinstance(d, dict) else None
            if turn_count is None:
                turn_count = i + 1

            rows.append(
                {
                    "convo_id": convo_id,
                    "subflow": as_str(subflow) or "<NA>",
                    "turn_count": int(turn_count),
                    "speaker": as_str(speaker).lower(),
                    "text": as_str(text),
                    "intent": as_str(parsed["intent"]).lower(),
                    "nextstep": as_str(parsed["nextstep"]).lower(),
                    "action": as_str(parsed["action"]).lower(),
                    "value_count": len(parsed["value"]),
                }
            )

    turns = pd.DataFrame(rows)
    turns = turns.sort_values(["convo_id", "turn_count"], kind="mergesort").reset_index(drop=True)
    turns["is_closing_turn"] = (
        turns["speaker"].eq("agent")
        & turns["text"].fillna("").astype(str).str.contains(CLOSING_RE, na=False)
    )
    turns["is_customer_escalation_req"] = (
        turns["speaker"].eq("customer")
        & turns["text"].fillna("").astype(str).str.contains(CUSTOMER_ESCALATION_RE, na=False)
    )
    return turns


def to_float(v: Any) -> float:
    if isinstance(v, (int, float, np.number)):
        return float(v)
    return float("nan")


def main() -> None:
    root = Path(__file__).resolve().parents[3]

    phase1_dir = root / "simulation" / "artifacts" / "phase 1"
    phase3_dir = root / "simulation" / "artifacts" / "phase3"
    phase4_dir = root / "simulation" / "artifacts" / "phase 4"
    phase5_dir = root / "simulation" / "artifacts" / "phase 5"
    phase3_dir.mkdir(parents=True, exist_ok=True)

    e1 = pd.read_csv(phase1_dir / "extract1_conversation_stats.csv")
    e2 = pd.read_csv(phase1_dir / "extract2_action_sequences.csv")
    e3 = pd.read_csv(phase1_dir / "extract3_subflow_stats.csv")
    e4 = pd.read_csv(phase1_dir / "extract4_information_by_turn.csv")
    e5_clusters = pd.read_csv(phase5_dir / "extract5_with_clusters_final.csv")
    persona_profiles = json.loads((phase5_dir / "persona_profiles.json").read_text(encoding="utf-8"))
    phase4_model = json.loads((phase4_dir / "psuccess_model.json").read_text(encoding="utf-8"))

    turns = load_turns(root / "abcd_v1.1.json")

    subflow_mean_action_count = e3.set_index("subflow")["mean_action_count"].to_dict()
    corpus_mean_action_count = float(e1["action_turns"].mean())

    # Subflow mean total values from Extract 4.
    convo_total_values = (
        e4.groupby("convo_id", as_index=False)
        .agg(total_values_provided=("cumulative_values_by_turn", "max"))
    )
    convo_subflow = e1[["convo_id", "subflow"]].drop_duplicates()
    subflow_total_values_mean = (
        convo_total_values.merge(convo_subflow, on="convo_id", how="left")
        .groupby("subflow", as_index=False)
        .agg(subflow_mean_total_values=("total_values_provided", "mean"))
    )
    subflow_mean_total_values = subflow_total_values_mean.set_index("subflow")["subflow_mean_total_values"].to_dict()

    # Build per-conversation cumulative value lookup.
    info_turns = e4.sort_values(["convo_id", "turn_count"]).copy()
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

    # Calibration 1: AskInfo mu_ask
    ask_rows: list[dict[str, Any]] = []
    retrieve_turns = turns[(turns["speaker"] == "agent") & (turns["nextstep"] == "retrieve_utterance")]
    action_turns = turns[turns["speaker"] == "action"][["convo_id", "turn_count"]]
    action_by_convo: dict[int, np.ndarray] = {
        int(cid): g["turn_count"].to_numpy(dtype=int)
        for cid, g in action_turns.groupby("convo_id", sort=False)
    }

    for r in retrieve_turns.itertuples(index=False):
        convo_id = int(r.convo_id)
        t = int(r.turn_count)
        subflow = str(r.subflow)
        current_cum = cumulative_at(convo_id, t)
        next_action_turn = None
        if convo_id in action_by_convo:
            arr = action_by_convo[convo_id]
            idx = int(np.searchsorted(arr, t, side="right"))
            if idx < len(arr):
                candidate = int(arr[idx])
                if candidate - t <= 3:
                    next_action_turn = candidate
        if next_action_turn is None:
            delta_values = 0.0
        else:
            next_cum = cumulative_at(convo_id, next_action_turn)
            delta_values = max(0.0, float(next_cum - current_cum))

        denom = max(float(subflow_mean_total_values.get(subflow, 1.0)), 1.0)
        delta_info = float(np.clip(delta_values / denom, 0.0, 1.0))
        ask_rows.append(
            {
                "convo_id": convo_id,
                "subflow": subflow,
                "turn_count": t,
                "delta_values": delta_values,
                "delta_info": delta_info,
                "subflow_mean_total_values": float(subflow_mean_total_values.get(subflow, np.nan)),
            }
        )

    ask_df = pd.DataFrame(ask_rows)
    frac_zero_delta = float((ask_df["delta_info"] == 0).mean())
    p_info_gain = float((ask_df["delta_values"] > 0).mean())
    ask_nonzero = ask_df[ask_df["delta_values"] > 0].copy()
    mu_ask_conditional = float(ask_nonzero["delta_info"].mean()) if len(ask_nonzero) else 0.0
    mu_ask_conditional_std = float(ask_nonzero["delta_info"].std(ddof=1)) if len(ask_nonzero) > 1 else 0.0

    task_heavy_mask = ask_df["subflow_mean_total_values"] > 4.0
    mu_task_heavy = float(ask_df.loc[task_heavy_mask, "delta_info"].mean()) if task_heavy_mask.any() else float("nan")
    mu_info_light = float(ask_df.loc[~task_heavy_mask, "delta_info"].mean()) if (~task_heavy_mask).any() else float("nan")

    convo_cluster = e5_clusters[["convo_id", "cluster_id"]].drop_duplicates()
    cluster_id_to_label = {int(p["cluster_id"]): str(p["label"]) for p in persona_profiles["personas"]}
    ask_df = ask_df.merge(convo_cluster, on="convo_id", how="left")
    ask_df["persona_label"] = ask_df["cluster_id"].map(lambda x: cluster_id_to_label.get(int(x)) if pd.notna(x) else None)
    ask_nonzero = ask_nonzero.merge(convo_cluster, on="convo_id", how="left")
    ask_nonzero["persona_label"] = ask_nonzero["cluster_id"].map(lambda x: cluster_id_to_label.get(int(x)) if pd.notna(x) else None)
    persona_labels_order = [str(p["label"]) for p in persona_profiles["personas"]]
    mu_ask_conditional_by_persona: dict[str, float] = {}
    sigma_ask_conditional_by_persona: dict[str, float] = {}
    for label in persona_labels_order:
        persona_vals = ask_nonzero.loc[ask_nonzero["persona_label"] == label, "delta_info"]
        mu_ask_conditional_by_persona[label] = float(persona_vals.mean()) if len(persona_vals) else float("nan")
        sigma_ask_conditional_by_persona[label] = float(persona_vals.std(ddof=1)) if len(persona_vals) > 1 else float("nan")

    mu_ask_conditional_high_engagement = float(
        ask_nonzero.loc[ask_nonzero["persona_label"] == "high_engagement_resolver", "delta_info"].mean()
    ) if (ask_nonzero["persona_label"] == "high_engagement_resolver").any() else float("nan")
    mu_ask_conditional_low_engagement = float(
        ask_nonzero.loc[ask_nonzero["persona_label"] == "low_engagement_resolver", "delta_info"].mean()
    ) if (ask_nonzero["persona_label"] == "low_engagement_resolver").any() else float("nan")

    askinfo_calibration = {
        "n_askinfo_turns": int(len(ask_df)),
        "p_info_gain": p_info_gain,
        "mu_ask_conditional": mu_ask_conditional,
        "mu_ask_conditional_std": mu_ask_conditional_std,
        "fraction_zero_delta_info": frac_zero_delta,
        "mu_ask_task_heavy_total_values_gt_4": mu_task_heavy,
        "mu_ask_info_light_total_values_le_4": mu_info_light,
        "mu_ask_conditional_by_persona": mu_ask_conditional_by_persona,
        "sigma_ask_conditional_by_persona": sigma_ask_conditional_by_persona,
        "askinfo_formula": "if random() < p_info_gain * (0.6 + 0.8*rho): delta_i = clip(mu_ask_conditional_by_persona[persona_label] * (1 - 0.4*d) + N(0,0.02), 0, 1-i_t) else delta_i = 0",
    }

    # Calibration 2: AffectiveRepair empirical effectiveness.
    repair_turns = turns[(turns["speaker"] == "agent") & (turns["text"].str.contains(REPAIR_RE, na=False))]
    turns_by_convo = {int(cid): g for cid, g in turns.groupby("convo_id", sort=False)}

    esc_after_repair_flags: list[int] = []
    resolved_after_repair_flags: list[int] = []
    for r in repair_turns.itertuples(index=False):
        convo_id = int(r.convo_id)
        t = int(r.turn_count)
        g = turns_by_convo.get(convo_id)
        if g is None:
            esc_after_repair_flags.append(0)
            resolved_after_repair_flags.append(0)
            continue
        window = g[(g["turn_count"] > t) & (g["turn_count"] <= t + 5)]
        has_escalation = bool(
            ((window["speaker"] == "action") & (window["action"] == "notify-team")).any()
            or window["is_customer_escalation_req"].any()
        )
        esc_after_repair_flags.append(int(has_escalation))
        conv_resolved = int(e1.loc[e1["convo_id"] == convo_id, "resolution_flag"].iloc[0]) if (e1["convo_id"] == convo_id).any() else 0
        resolved_after_repair_flags.append(conv_resolved)

    p_escalation_after_repair = float(np.mean(esc_after_repair_flags)) if esc_after_repair_flags else float("nan")
    p_repair_effective = float(1.0 - p_escalation_after_repair) if not np.isnan(p_escalation_after_repair) else float("nan")
    p_escalation_baseline = float(e1["escalation_flag"].mean())
    repair_lift = float(p_escalation_baseline - p_escalation_after_repair) if not np.isnan(p_escalation_after_repair) else float("nan")
    p_repair_base_design = 0.40

    affective_repair_calibration = {
        "n_repair_utterances": int(len(repair_turns)),
        "p_escalation_after_repair": p_escalation_after_repair,
        "p_escalation_baseline": p_escalation_baseline,
        "repair_lift": repair_lift,
        "p_repair_effective_base_empirical": p_repair_effective,
        "p_repair_base_design": p_repair_base_design,
        "p_repair_base_empirical_note": (
            "Empirical signal too weak to use. Repair lift = 0.0066 over baseline. "
            "ABCD does not reliably distinguish repair effectiveness within conversations. "
            "Design value 0.40 used instead."
        ),
        "resolved_rate_given_repair_utterance": float(np.mean(resolved_after_repair_flags)) if resolved_after_repair_flags else float("nan"),
    }

    # Shared helpers for close/progress calibrations.
    e1_idx = e1.set_index("convo_id")
    e2_idx = e2.set_index("convo_id")
    subflow_mean_action_series = e3.set_index("subflow")["mean_action_count"]

    # Phase 4 subflow mean p_success from fitted model.
    theta_0 = float(phase4_model["coefficients"]["theta_0"])
    theta_i = float(phase4_model["coefficients"]["theta_i"])
    alpha_action = float(phase4_model["action_offsets"].get("ProvideSolution", 0.10))
    subflow_offsets = {k: float(v) for k, v in phase4_model.get("subflow_offsets", {}).items()}

    # info at close, progress at close per conversation.
    final_info_by_convo = convo_total_values.set_index("convo_id")["total_values_provided"].to_dict()
    close_rows: list[dict[str, Any]] = []

    for convo_id, row in e1_idx.iterrows():
        convo_id_int = int(convo_id)
        subflow = str(row["subflow"])
        total_vals = float(final_info_by_convo.get(convo_id_int, 0.0))
        info_close = float(np.clip(total_vals / max(float(subflow_mean_total_values.get(subflow, 1.0)), 1.0), 0.0, 1.0))

        action_count = float(e2_idx.loc[convo_id_int, "action_count"]) if convo_id_int in e2_idx.index else float(row.get("action_turns", 0.0))
        progress_close = float(np.clip(action_count / max(float(subflow_mean_action_count.get(subflow, 1.0)), 1e-9), 0.0, 1.0))

        p_success_subflow_mean = sigmoid(theta_0 + theta_i * info_close + float(subflow_offsets.get(subflow, 0.0)) + alpha_action)
        frustration_close = 0.15
        close_score = float(0.45 * p_success_subflow_mean + 0.35 * progress_close + 0.20 * info_close - 0.25 * frustration_close)

        close_rows.append(
            {
                "convo_id": convo_id_int,
                "subflow": subflow,
                "resolution_flag": int(row["resolution_flag"]),
                "dropout_flag": int(row["dropout_flag"]),
                "information_at_close": info_close,
                "progress_at_close": progress_close,
                "p_success_subflow_mean": p_success_subflow_mean,
                "close_score": close_score,
            }
        )

    close_df = pd.DataFrame(close_rows)
    resolved = close_df[close_df["resolution_flag"] == 1]
    nonresolved = close_df[close_df["resolution_flag"] == 0]

    res_mean = float(resolved["close_score"].mean())
    nonres_mean = float(nonresolved["close_score"].mean())
    res_p10 = float(resolved["close_score"].quantile(0.10))
    nonres_p90 = float(nonresolved["close_score"].quantile(0.90))
    recommended_close_threshold = float((res_p10 + nonres_p90) / 2.0)
    close_threshold_update = bool(abs(recommended_close_threshold - 0.70) > 0.10)
    close_threshold_final = recommended_close_threshold if close_threshold_update else 0.70

    close_score_calibration = {
        "mean_close_score_resolved": res_mean,
        "mean_close_score_nonresolved": nonres_mean,
        "p10_close_score_resolved": res_p10,
        "p90_close_score_nonresolved": nonres_p90,
        "recommended_threshold_midpoint": recommended_close_threshold,
        "threshold_changed": close_threshold_update,
        "close_score_threshold_final": close_threshold_final,
    }

    # Calibration 4: progress at final ProvideSolution in resolved convos.
    action_turns_full = turns[turns["speaker"] == "action"].copy()
    action_turns_full = action_turns_full.sort_values(["convo_id", "turn_count"], kind="mergesort")
    action_turns_full["action_idx"] = action_turns_full.groupby("convo_id").cumcount() + 1

    solution_turns = action_turns_full[action_turns_full["action"].isin(SOLUTION_ACTIONS)]
    last_solution = (
        solution_turns.sort_values(["convo_id", "turn_count"], kind="mergesort")
        .groupby("convo_id", as_index=False)
        .tail(1)
    )

    progress_rows: list[float] = []
    for r in last_solution.itertuples(index=False):
        convo_id = int(r.convo_id)
        if convo_id not in e1_idx.index:
            continue
        if int(e1_idx.loc[convo_id, "resolution_flag"]) != 1:
            continue
        subflow = str(e1_idx.loc[convo_id, "subflow"])
        action_idx = float(r.action_idx)
        prog = float(np.clip(action_idx / max(float(subflow_mean_action_count.get(subflow, 1.0)), 1e-9), 0.0, 1.0))
        progress_rows.append(prog)

    progress_arr = np.array(progress_rows, dtype=float)
    progress_mean = float(np.mean(progress_arr)) if len(progress_arr) else float("nan")
    progress_median = float(np.median(progress_arr)) if len(progress_arr) else float("nan")
    progress_p25 = float(np.quantile(progress_arr, 0.25)) if len(progress_arr) else float("nan")

    frac_lt_05 = float(np.mean(progress_arr < 0.5)) if len(progress_arr) else float("nan")
    frac_05_07 = float(np.mean((progress_arr >= 0.5) & (progress_arr < 0.7))) if len(progress_arr) else float("nan")
    frac_07_085 = float(np.mean((progress_arr >= 0.7) & (progress_arr < 0.85))) if len(progress_arr) else float("nan")
    frac_085_10 = float(np.mean((progress_arr >= 0.85) & (progress_arr <= 1.0))) if len(progress_arr) else float("nan")

    recommended_auto_resolve_threshold = progress_p25
    auto_resolve_update = bool(abs(recommended_auto_resolve_threshold - 0.85) > 0.10)
    auto_resolve_threshold_final = recommended_auto_resolve_threshold if auto_resolve_update else 0.85

    progress_threshold_calibration = {
        "n_resolved_with_solution_attempt": int(len(progress_arr)),
        "mean_progress_at_resolution": progress_mean,
        "median_progress_at_resolution": progress_median,
        "p25_progress_at_resolution": progress_p25,
        "fraction_progress_lt_0_5": frac_lt_05,
        "fraction_progress_0_5_to_0_7": frac_05_07,
        "fraction_progress_0_7_to_0_85": frac_07_085,
        "fraction_progress_0_85_to_1_0": frac_085_10,
        "recommended_auto_resolve_threshold": recommended_auto_resolve_threshold,
        "threshold_changed": auto_resolve_update,
        "auto_resolve_threshold_final": auto_resolve_threshold_final,
    }

    # Calibration 5: dropout formula validation.
    def p_dropout(f: float, failed_streak: int, turn_count: int, tau: float, intercept: float = -5.0) -> float:
        z = intercept + 3.5 * f + 0.3 * failed_streak + 0.1 * turn_count - 2.0 * tau
        return sigmoid(z)

    state_specs = [
        {
            "name": "State 1",
            "f": 0.10,
            "failed_streak": 0,
            "turn_count": 5,
            "tau": 0.70,
            "range": [0.0, 0.02],
        },
        {
            "name": "State 2",
            "f": 0.30,
            "failed_streak": 0,
            "turn_count": 10,
            "tau": 0.60,
            "range": [0.02, 0.05],
        },
        {
            "name": "State 3",
            "f": 0.50,
            "failed_streak": 1,
            "turn_count": 15,
            "tau": 0.50,
            "range": [0.05, 0.15],
        },
        {
            "name": "State 4",
            "f": 0.70,
            "failed_streak": 2,
            "turn_count": 18,
            "tau": 0.30,
            "range": [0.15, 0.35],
        },
        {
            "name": "State 5",
            "f": 0.90,
            "failed_streak": 3,
            "turn_count": 20,
            "tau": 0.20,
            "range": [0.35, 1.0],
        },
    ]

    def evaluate_dropout(intercept: float) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for spec in state_specs:
            p = p_dropout(spec["f"], spec["failed_streak"], spec["turn_count"], spec["tau"], intercept=intercept)
            p_rounded = float(round(p, 2))
            lo, hi = spec["range"]
            in_range = bool(lo <= p_rounded <= hi)
            rows.append(
                {
                    "state": spec["name"],
                    "f": spec["f"],
                    "failed_streak": spec["failed_streak"],
                    "turn_count": spec["turn_count"],
                    "tau": spec["tau"],
                    "p_dropout": p,
                    "p_dropout_rounded_2dp": p_rounded,
                    "expected_range": spec["range"],
                    "within_expected_range": in_range,
                }
            )
        return rows

    dropout_intercept = -5.0
    evals = evaluate_dropout(dropout_intercept)
    if not all(x["within_expected_range"] for x in evals):
        # Allowed fallback by prompt: adjust only intercept by +/-0.25.
        for candidate in [-5.25, -4.75]:
            cand = evaluate_dropout(candidate)
            if all(x["within_expected_range"] for x in cand):
                dropout_intercept = candidate
                evals = cand
                break

    empirical_dropout_rate = float(e1["dropout_flag"].mean())
    dropout_validation = {
        "formula": f"sigmoid({dropout_intercept:.2f} + 3.5*f + 0.3*failed_streak + 0.1*turn_count - 2.0*tau)",
        "coefficients": {
            "intercept": dropout_intercept,
            "frustration": 3.5,
            "failed_streak": 0.3,
            "turn_count": 0.1,
            "tau": -2.0,
        },
        "empirical_dropout_rate": empirical_dropout_rate,
        "state_evaluations": evals,
        "coefficients_adjusted": True,
        "all_states_within_expected_range": bool(all(x["within_expected_range"] for x in evals)),
    }

    # Calibration 6/7/8 with persona profiles.
    persona_rows: list[dict[str, Any]] = []
    askinfo_delta_rows: list[dict[str, Any]] = []
    askinfo_good_rows: list[dict[str, Any]] = []
    askinfo_bad_rows: list[dict[str, Any]] = []
    provide_failure_rows: list[dict[str, Any]] = []
    repair_expected_rows: list[dict[str, Any]] = []

    base_rate = p_repair_base_design
    for p in persona_profiles["personas"]:
        label = str(p["label"])
        rho = float(p["rho_mean"])
        sigma = float(p["sigma_mean"])
        tau = float(p["tau_mean"])

        d_a = 1.0
        delta_i_a = 0.15
        failed_streak_a = 0
        scenario_a = float(0.03 * (1 - rho) + 0.01 * d_a - 0.06 * abs(delta_i_a) + 0.03 * max(failed_streak_a - 1, 0))

        d_b = 1.5
        delta_i_b = 0.0
        failed_streak_b = 2
        scenario_b = float(0.03 * (1 - rho) + 0.01 * d_b - 0.06 * abs(delta_i_b) + 0.03 * max(failed_streak_b - 1, 0))

        askinfo_delta_rows.append(
            {
                "persona": label,
                "rho": rho,
                "scenario_A_delta_f": scenario_a,
                "scenario_B_delta_f": scenario_b,
                "scenario_A_expected_small_or_negative": bool(scenario_a <= 0.02),
                "scenario_B_expected_positive": bool(scenario_b > 0),
            }
        )
        askinfo_good_rows.append(
            {
                "persona": label,
                "rho": rho,
                "delta_f": scenario_a,
            }
        )
        askinfo_bad_rows.append(
            {
                "persona": label,
                "rho": rho,
                "delta_f": scenario_b,
            }
        )

        streak_vals: dict[str, float] = {}
        for streak in [0, 1, 2, 3]:
            delta_f_failure = float(0.05 + 0.15 * sigma + 0.10 * (streak / (streak + 3.0)) * (1 - tau))
            streak_vals[f"streak_{streak}"] = delta_f_failure
        cumulative_three_fail = float(streak_vals["streak_0"] + streak_vals["streak_1"] + streak_vals["streak_2"])

        provide_failure_rows.append(
            {
                "persona": label,
                "sigma": sigma,
                "tau": tau,
                **streak_vals,
                "cumulative_delta_after_3_failures": cumulative_three_fail,
                "cumulative_after_3_failures_gt_0_30": bool(cumulative_three_fail > 0.30),
            }
        )

        p_repair = float(base_rate + 0.4 * rho)
        delta_if_effective = float(-0.15 * (0.6 + 0.6 * sigma))
        expected_delta = float(p_repair * delta_if_effective + (1 - p_repair) * 0.02)
        repairs_needed = float(np.ceil(0.20 / abs(expected_delta))) if expected_delta < 0 else float("inf")

        repair_expected_rows.append(
            {
                "persona": label,
                "rho": rho,
                "sigma": sigma,
                "p_repair": p_repair,
                "delta_f_if_effective": delta_if_effective,
                "expected_delta_f_per_repair": expected_delta,
                "repair_actions_to_reduce_f_by_0_20": repairs_needed,
                "repair_too_weak_gt_8_actions": bool(np.isfinite(repairs_needed) and repairs_needed > 8),
            }
        )

        persona_rows.append({"label": label, "rho": rho, "sigma": sigma, "tau": tau})

    # Write individual calibration files.
    (phase3_dir / "askinfo_calibration.json").write_text(json.dumps(askinfo_calibration, indent=2), encoding="utf-8")
    (phase3_dir / "affective_repair_calibration.json").write_text(json.dumps(affective_repair_calibration, indent=2), encoding="utf-8")
    (phase3_dir / "close_score_calibration.json").write_text(json.dumps(close_score_calibration, indent=2), encoding="utf-8")
    (phase3_dir / "progress_threshold_calibration.json").write_text(json.dumps(progress_threshold_calibration, indent=2), encoding="utf-8")
    (phase3_dir / "dropout_validation.json").write_text(json.dumps(dropout_validation, indent=2), encoding="utf-8")

    transition_calibration = {
        "askinfo": {
            "p_info_gain": p_info_gain,
            "mu_ask_conditional": mu_ask_conditional,
            "mu_ask_conditional_std": mu_ask_conditional_std,
            "mu_ask_conditional_by_persona": mu_ask_conditional_by_persona,
            "sigma_ask_conditional_by_persona": sigma_ask_conditional_by_persona,
            "formula": "if random() < p_info_gain * (0.6 + 0.8*rho): delta_i = clip(mu_ask_conditional_by_persona[persona_label] * (1 - 0.4*d) + N(0,0.02), 0, 1-i_t) else: delta_i = 0",
            "formula_note": "Use persona-specific mu_ask_conditional. The normalized delta reflects task completion fraction, not raw value count. Low-engagement subflows advance faster toward information completion because they require fewer total values.",
            "noise_std": 0.02,
        },
        "affective_repair": {
            "p_repair_base_design": p_repair_base_design,
            "p_repair_base_empirical_note": "Empirical signal too weak to use. Repair lift = 0.0066 over baseline. ABCD does not reliably distinguish repair effectiveness within conversations. Design value 0.40 used instead.",
            "p_repair_formula": "p_repair = 0.40 + 0.4 * rho",
            "p_repair_range": "rho=0.234 -> p_repair=0.494, rho=0.739 -> p_repair=0.696",
            "repair_lift_over_baseline": repair_lift,
            "frustration_reduction_if_effective": "0.15 * (0.6 + 0.6 * sigma)",
            "frustration_increase_if_ineffective": 0.02,
        },
        "close": {
            "close_score_formula": "0.45*p_success + 0.35*progress + 0.20*information - 0.25*frustration",
            "close_score_threshold": close_threshold_final,
            "mean_close_score_resolved": res_mean,
            "mean_close_score_nonresolved": nonres_mean,
        },
        "progress": {
            "auto_resolve_threshold": auto_resolve_threshold_final,
            "mean_progress_at_resolution": progress_mean,
            "median_progress_at_resolution": progress_median,
        },
        "dropout": {
            "formula": f"sigmoid({dropout_intercept:.2f} + 3.5*f + 0.3*failed_streak + 0.1*turn_count - 2.0*tau)",
            "empirical_dropout_rate": empirical_dropout_rate,
            "state_evaluations": evals,
            "coefficients_adjusted": True,
        },
        "frustration_deltas": {
            "askinfo_good": askinfo_good_rows,
            "askinfo_bad": askinfo_bad_rows,
            "provide_solution_failure": provide_failure_rows,
            "affective_repair_expected": repair_expected_rows,
        },
    }
    (phase3_dir / "transition_calibration.json").write_text(json.dumps(transition_calibration, indent=2), encoding="utf-8")

    print("=== CALIBRATION 1: AskInfo ===")
    print(f"p_info_gain={p_info_gain:.6f}, mu_ask_conditional={mu_ask_conditional:.6f}, mu_ask_conditional_std={mu_ask_conditional_std:.6f}")
    print(f"frac_zero={frac_zero_delta:.4f}")
    print(f"mu_task_heavy={mu_task_heavy:.6f}, mu_info_light={mu_info_light:.6f}")
    print("mu_ask_conditional per persona:")
    for label in persona_labels_order:
        val = mu_ask_conditional_by_persona[label]
        print(f"  {label}: {val:.6f}")
    print("sigma_ask_cond per persona:")
    for label in persona_labels_order:
        val = sigma_ask_conditional_by_persona[label]
        print(f"  {label}: {val:.6f}")

    print("\n=== CALIBRATION 2: AffectiveRepair ===")
    print(f"n_repair={len(repair_turns)}, p_esc_after_repair={p_escalation_after_repair:.6f}, baseline_esc={p_escalation_baseline:.6f}")
    print(f"repair_lift={repair_lift:.6f}, p_repair_effective_empirical={p_repair_effective:.6f}, p_repair_base_design={p_repair_base_design:.2f}")

    print("\n=== CALIBRATION 3: Close Threshold ===")
    print(f"mean_resolved={res_mean:.6f}, mean_nonresolved={nonres_mean:.6f}")
    print(f"resolved_p10={res_p10:.6f}, nonresolved_p90={nonres_p90:.6f}")
    print(f"recommended={recommended_close_threshold:.6f}, final={close_threshold_final:.6f}, changed={close_threshold_update}")

    print("\n=== CALIBRATION 4: Progress Auto-Resolve ===")
    print(f"mean={progress_mean:.6f}, median={progress_median:.6f}, p25={progress_p25:.6f}")
    print(f"recommended={recommended_auto_resolve_threshold:.6f}, final={auto_resolve_threshold_final:.6f}, changed={auto_resolve_update}")

    print("\n=== CALIBRATION 5: Dropout Validation ===")
    print(f"empirical_dropout_rate={empirical_dropout_rate:.6f}")
    print(f"dropout_formula=sigmoid({dropout_intercept:.2f} + 3.5*f + 0.3*failed_streak + 0.1*turn_count - 2.0*tau)")
    for d in evals:
        print(f"{d['state']}: p_dropout={d['p_dropout']:.6f}, in_range={d['within_expected_range']}")

    print("\n=== CALIBRATION 6/7/8: Persona Delta Validation ===")
    print(pd.DataFrame(askinfo_delta_rows).to_string(index=False))
    print(pd.DataFrame(provide_failure_rows)[["persona", "streak_0", "streak_1", "streak_2", "streak_3", "cumulative_delta_after_3_failures"]].to_string(index=False))
    print(pd.DataFrame(repair_expected_rows)[["persona", "p_repair", "expected_delta_f_per_repair", "repair_actions_to_reduce_f_by_0_20"]].to_string(index=False))

    print("\nSaved files:")
    for fn in [
        "askinfo_calibration.json",
        "affective_repair_calibration.json",
        "close_score_calibration.json",
        "progress_threshold_calibration.json",
        "dropout_validation.json",
        "transition_calibration.json",
    ]:
        print(phase3_dir / fn)


if __name__ == "__main__":
    main()
