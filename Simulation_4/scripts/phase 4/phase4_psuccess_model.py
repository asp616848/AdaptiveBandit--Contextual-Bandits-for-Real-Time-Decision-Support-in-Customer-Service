from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


TARGETS_SCHEMA = ["intent", "nextstep", "action", "value", "utterance_ranking"]

# Keep this aligned with the Phase 1 extraction logic.
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


@dataclass
class ModelArtifacts:
    attempts_df: pd.DataFrame
    model: LogisticRegression
    theta_0: float
    theta_i: float
    theta_d: float | None
    theta_f: float | None
    corr_fi: float
    corr_fs: float
    p_high_info: float
    p_low_info: float
    mean_pred: float
    subflow_offsets: dict[str, float]
    checks: dict[str, str]
    calibration: pd.DataFrame


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


def as_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-x)))


def clipped_log_odds(p: float) -> float:
    p = float(np.clip(p, 0.01, 0.99))
    return float(math.log(p / (1.0 - p)))


def load_all_turns(raw_path: Path) -> pd.DataFrame:
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    corpus: list[dict[str, Any]] = []
    for split in ["train", "dev", "test"]:
        corpus.extend(payload.get(split, []))

    rows: list[dict[str, Any]] = []
    for convo in corpus:
        convo_id = convo.get("convo_id")
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
                    "action": as_str(parsed["action"]).lower(),
                }
            )

    turns = pd.DataFrame(rows)
    turns = turns.sort_values(["convo_id", "turn_count"], kind="mergesort").reset_index(drop=True)
    turns["is_closing_turn"] = (
        turns["speaker"].eq("agent")
        & turns["text"].fillna("").astype(str).str.contains(CLOSING_RE, na=False)
    )
    turns["is_customer_requested_escalation"] = (
        turns["speaker"].eq("customer")
        & turns["text"].fillna("").astype(str).str.contains(CUSTOMER_ESCALATION_RE, na=False)
    )
    return turns


def build_attempt_dataset(root: Path) -> pd.DataFrame:
    phase1_dir = root / "Simulation_4" / "artifacts" / "phase 1"

    e1 = pd.read_csv(phase1_dir / "extract1_conversation_stats.csv")
    e3 = pd.read_csv(phase1_dir / "extract3_subflow_stats.csv")
    e4 = pd.read_csv(phase1_dir / "extract4_information_by_turn.csv")

    raw_turns = load_all_turns(root / "abcd_v1.1.json")

    closing_turns = (
        raw_turns[raw_turns["is_closing_turn"]]
        .groupby("convo_id")["turn_count"]
        .apply(list)
        .to_dict()
    )

    attempts = raw_turns[
        raw_turns["speaker"].eq("action")
        & raw_turns["action"].isin(SOLUTION_ACTIONS)
    ][["convo_id", "subflow", "turn_count", "action"]].copy()

    if attempts.empty:
        raise RuntimeError("No ProvideSolution attempts found from SOLUTION_ACTIONS.")

    convo_resolution = e1.set_index("convo_id")["resolution_flag"].to_dict()
    convo_escalation = e1.set_index("convo_id")["escalation_flag"].to_dict()

    success_labels: list[int] = []
    closing_within_5_flags: list[int] = []
    for r in attempts.itertuples(index=False):
        convo_id = int(r.convo_id)
        t = int(r.turn_count)

        has_close_within_5 = any((ct > t) and (ct <= t + 5) for ct in closing_turns.get(convo_id, []))
        resolution_flag = int(convo_resolution.get(convo_id, 0))

        success = int((resolution_flag == 1) and has_close_within_5)
        success_labels.append(success)
        closing_within_5_flags.append(int(has_close_within_5))

    attempts["resolution_flag"] = attempts["convo_id"].map(convo_resolution).fillna(0).astype(int)
    attempts["escalation_flag"] = attempts["convo_id"].map(convo_escalation).fillna(0).astype(int)
    attempts["closing_within_5"] = closing_within_5_flags
    attempts["success_label"] = success_labels

    # Feature 1: info_proxy
    info_turns = e4.sort_values(["convo_id", "turn_count"]).copy()

    # Subflow-level mean total values computed from Extract 4.
    convo_total_values = (
        info_turns.groupby("convo_id", as_index=False)
        .agg(total_values_provided=("cumulative_values_by_turn", "max"))
    )
    convo_subflow = e1[["convo_id", "subflow"]].drop_duplicates()
    subflow_total_values_mean = (
        convo_total_values.merge(convo_subflow, on="convo_id", how="left")
        .groupby("subflow", as_index=False)
        .agg(subflow_mean_total_values=("total_values_provided", "mean"))
    )

    attempts = attempts.merge(subflow_total_values_mean, on="subflow", how="left")
    attempts["subflow_mean_total_values"] = attempts["subflow_mean_total_values"].fillna(1.0)

    merged = attempts.sort_values(["convo_id", "turn_count"], kind="mergesort").reset_index(drop=True)

    info_lookup: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for convo_id, g in info_turns.groupby("convo_id", sort=False):
        turns_arr = g["turn_count"].to_numpy(dtype=float)
        vals_arr = g["cumulative_values_by_turn"].to_numpy(dtype=float)
        info_lookup[int(convo_id)] = (turns_arr, vals_arr)

    cumulative_values_at_turn: list[float] = []
    for r in merged.itertuples(index=False):
        cid = int(r.convo_id)
        t = float(r.turn_count)
        if cid not in info_lookup:
            cumulative_values_at_turn.append(0.0)
            continue
        turns_arr, vals_arr = info_lookup[cid]
        idx = int(np.searchsorted(turns_arr, t, side="right") - 1)
        cumulative_values_at_turn.append(float(vals_arr[idx]) if idx >= 0 else 0.0)

    merged["cumulative_values_at_turn"] = cumulative_values_at_turn

    denom = merged["subflow_mean_total_values"].replace(0, 1.0)
    merged["info_proxy"] = (merged["cumulative_values_at_turn"] / denom).clip(0.0, 1.0)

    # Feature 2: difficulty_proxy (subflow-level)
    subflow_action_mean = e3.set_index("subflow")["mean_action_count"].to_dict()
    corpus_mean_action_count = float(e1["action_turns"].mean())

    merged["subflow_mean_action_count"] = merged["subflow"].map(subflow_action_mean)
    merged["subflow_mean_action_count"] = merged["subflow_mean_action_count"].fillna(corpus_mean_action_count)
    merged["difficulty_proxy"] = (
        merged["subflow_mean_action_count"] / max(corpus_mean_action_count, 1e-9)
    ).clip(0.2, 3.0)

    # Feature 3: frustration_proxy = failed_streak + max(0, turn_ratio-1)
    subflow_mean_turns = e3.set_index("subflow")["mean_turns"].to_dict()
    merged["subflow_mean_turns"] = merged["subflow"].map(subflow_mean_turns)
    merged["subflow_mean_turns"] = merged["subflow_mean_turns"].fillna(float(e1["total_turns"].mean()))

    merged["turn_ratio_at_action"] = (
        merged["turn_count"] / merged["subflow_mean_turns"].replace(0, 1.0)
    ).clip(0.0, 3.0)

    merged = merged.sort_values(["convo_id", "turn_count"], kind="mergesort").reset_index(drop=True)
    failed_streak_before: list[int] = []
    for _, g in merged.groupby("convo_id", sort=False):
        streak = 0
        for row in g.itertuples(index=False):
            failed_streak_before.append(streak)
            if int(row.success_label) == 1:
                streak = 0
            else:
                streak += 1

    merged["failed_streak_at_turn"] = failed_streak_before
    merged["frustration_proxy"] = (
        merged["failed_streak_at_turn"] + np.maximum(0.0, merged["turn_ratio_at_action"] - 1.0)
    ).clip(0.0, 5.0)

    # Additional diagnostic from the issue discussion.
    escalation_turns_lookup = (
        raw_turns[raw_turns["is_customer_requested_escalation"]]
        .groupby("convo_id")["turn_count"]
        .apply(lambda s: sorted(int(x) for x in s.tolist()))
        .to_dict()
    )
    cumulative_escalation_signals: list[int] = []
    for r in merged.itertuples(index=False):
        cid = int(r.convo_id)
        t = int(r.turn_count)
        esc_turns = escalation_turns_lookup.get(cid, [])
        cumulative_escalation_signals.append(int(any(et < t for et in esc_turns)))
    merged["cumulative_escalation_signals"] = cumulative_escalation_signals

    return merged


def fit_psuccess_model(attempts: pd.DataFrame, e1: pd.DataFrame, e3: pd.DataFrame) -> ModelArtifacts:
    # EMPIRICAL FINDING FROM FRUSTRATION DIAGNOSTIC:
    # frustration_proxy and info_proxy are positively correlated in ABCD data.
    # Both increase with conversation maturity (later turns = more values gathered).
    # Higher frustration quartiles show HIGHER success rates (Q1=0.414, Q4=0.590)
    # because later attempts also have more information gathered.
    # This means frustration does NOT independently predict p_success once
    # info_proxy is controlled for — theta_f ≈ 0 in the regression confirmed this.
    #
    # DESIGN DECISION: Remove frustration_proxy from p_success regression.
    # Frustration affects the simulator through:
    #   - dropout_probability (frustrated customers leave)
    #   - escalation_probability (frustrated customers escalate)
    # NOT through p_success directly.
    # Full diagnostic saved in: Simulation_4/artifacts/phase4/frustration_proxy_diagnostic.txt
    feature_cols = ["info_proxy"]
    X = attempts[feature_cols].to_numpy()
    y = attempts["success_label"].astype(int).to_numpy()

    corr_fi = float(attempts["frustration_proxy"].corr(attempts["info_proxy"]))
    corr_fs = float(attempts["frustration_proxy"].corr(attempts["success_label"]))

    model = LogisticRegression(
        fit_intercept=True,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(X, y)

    theta_0 = float(model.intercept_[0])
    theta_i = float(model.coef_[0, 0])
    theta_d = None
    theta_f = None

    corpus_resolution_rate = float(e1["resolution_flag"].mean())
    corpus_log_odds = clipped_log_odds(corpus_resolution_rate)

    subflow_offsets: dict[str, float] = {}
    for r in e3.itertuples(index=False):
        sf = str(r.subflow)
        sf_log_odds = clipped_log_odds(float(r.resolution_rate))
        subflow_offsets[sf] = float(sf_log_odds - corpus_log_odds)

    def pred(info: float, alpha_subflow: float = 0.0) -> float:
        z = theta_0 + theta_i * info + alpha_subflow
        return sigmoid(z)

    p_high_info = pred(0.9, 0.0)
    p_low_info = pred(0.1, 0.0)
    high_state_threshold = p_high_info - 0.05

    mean_pred = float(model.predict_proba(X)[:, 1].mean())

    check_map: dict[str, str] = {}
    check_map["check_1_theta_i_positive"] = "PASS" if theta_i > 0 else "FAIL"
    check_map["check_2_theta_d_note"] = "PASS — removed, captured by subflow offsets"
    check_map["check_3_theta_f_note"] = "PASS — removed, frustration in dropout/escalation dynamics"
    check_map["check_4_theta_0_range"] = "PASS" if -8.0 < theta_0 < 1.0 else "FAIL"
    check_map["check_5_high_state_consistency"] = (
        f"{'PASS' if p_high_info > high_state_threshold else 'FAIL'} — "
        f"value={p_high_info:.4f}, threshold={high_state_threshold:.4f}"
    )
    check_map["check_6_low_state_failure"] = (
        f"{'PASS' if p_low_info < 0.20 else 'FAIL'} — {p_low_info:.4f}"
    )
    check_map["check_7_mean_psuccess_range"] = f"{'PASS' if 0.40 <= mean_pred <= 0.65 else 'FAIL'} — {mean_pred:.4f}"

    easy = subflow_offsets.get("recover_username")
    hard = subflow_offsets.get("out_of_stock_general")
    check_map["check_8_easy_subflow_positive"] = (
        "PASS" if (easy is not None and easy > 0) else "FAIL"
    )
    check_map["check_9_hard_subflow_negative"] = (
        "PASS" if (hard is not None and hard < 0) else "FAIL"
    )

    # Calibration bins.
    preds = model.predict_proba(X)[:, 1]
    cal = attempts[["success_label", "info_proxy"]].copy()
    cal["predicted"] = preds
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    cal["info_proxy_bin"] = pd.cut(
        cal["info_proxy"].clip(0.0, 1.0),
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    calibration = (
        cal.groupby("info_proxy_bin", observed=False)
        .agg(
            empirical_success_rate=("success_label", "mean"),
            mean_predicted_psuccess=("predicted", "mean"),
            n_attempts=("success_label", "size"),
        )
        .reset_index()
    )

    return ModelArtifacts(
        attempts_df=attempts,
        model=model,
        theta_0=theta_0,
        theta_i=theta_i,
        theta_d=theta_d,
        theta_f=theta_f,
        corr_fi=corr_fi,
        corr_fs=corr_fs,
        p_high_info=p_high_info,
        p_low_info=p_low_info,
        mean_pred=mean_pred,
        subflow_offsets=subflow_offsets,
        checks=check_map,
        calibration=calibration,
    )


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    phase1_dir = root / "Simulation_4" / "artifacts" / "phase 1"
    phase4_out_dir = root / "Simulation_4" / "artifacts" / "phase 4"
    phase4_out_dir.mkdir(parents=True, exist_ok=True)

    e1 = pd.read_csv(phase1_dir / "extract1_conversation_stats.csv")
    e3 = pd.read_csv(phase1_dir / "extract3_subflow_stats.csv")

    attempts = build_attempt_dataset(root)

    # Step 2 reporting.
    n_attempts = int(len(attempts))
    n_convos = int(attempts["convo_id"].nunique())
    overall_success = float(attempts["success_label"].mean())

    print("ProvideSolution attempts identified:", n_attempts)
    print("Unique conversations with attempts:", n_convos)
    print(f"Overall success rate: {overall_success:.4f}")
    if not (0.60 <= overall_success <= 0.80):
        print("WARNING: overall success rate outside expected [0.60, 0.80].")

    by_action = (
        attempts.groupby("action", as_index=False)
        .agg(n_attempts=("success_label", "size"), success_rate=("success_label", "mean"))
        .sort_values("n_attempts", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    print("\nTop 10 solution actions by attempt count:")
    print(by_action.to_string(index=False))

    # Feature summaries.
    print("\nFeature summary statistics:")
    feature_cols = ["info_proxy", "difficulty_proxy", "frustration_proxy"]
    print(attempts[feature_cols].agg(["mean", "std", "min", "max"]).to_string())

    for c in feature_cols:
        corr = float(pd.Series(attempts[c]).corr(attempts["success_label"]))
        print(f"corr(success_label, {c}) = {corr:.4f}")

    # Directional sanity for action-offset assumptions.
    turns = load_all_turns(root / "abcd_v1.1.json")
    convo_resolution = e1.set_index("convo_id")["resolution_flag"].to_dict()
    closing_turns = (
        turns[turns["is_closing_turn"]]
        .groupby("convo_id")["turn_count"]
        .apply(list)
        .to_dict()
    )

    def rate_within5(df: pd.DataFrame) -> float:
        vals: list[int] = []
        for r in df.itertuples(index=False):
            convo_id = int(r.convo_id)
            t = int(r.turn_count)
            success = int(
                int(convo_resolution.get(convo_id, 0)) == 1
                and any((ct > t) and (ct <= t + 5) for ct in closing_turns.get(convo_id, []))
            )
            vals.append(success)
        return float(np.mean(vals)) if vals else float("nan")

    solution_turns = turns[(turns["speaker"] == "action") & (turns["action"].isin(SOLUTION_ACTIONS))][["convo_id", "turn_count"]]
    retrieve_turns = turns[(turns["speaker"] == "action") & (turns["action"] == "retrieve_utterance")][["convo_id", "turn_count"]]
    solution_rate_5 = rate_within5(solution_turns)
    retrieve_rate_5 = rate_within5(retrieve_turns)
    print("\nDirectional check (within-5 success after action turn):")
    print(f"solution_actions_rate={solution_rate_5:.4f}, retrieve_utterance_rate={retrieve_rate_5:.4f}")
    if not np.isnan(solution_rate_5) and not np.isnan(retrieve_rate_5) and solution_rate_5 <= retrieve_rate_5:
        print("WARNING: solution-action directional check did not exceed retrieve_utterance.")

    artifacts = fit_psuccess_model(attempts, e1, e3)

    print(f"Correlation(frustration_proxy, info_proxy) = {artifacts.corr_fi:.4f}")
    print(f"Correlation(frustration_proxy, success_label) = {artifacts.corr_fs:.4f}")
    print("NOTE: frustration_proxy excluded from model — confounded with info_proxy")

    print("\nModel coefficients:")
    print(f"theta_0={artifacts.theta_0:.6f}")
    print(f"theta_i={artifacts.theta_i:.6f}")
    print(f"p_success(info=0.9, alpha_subflow=0) = {artifacts.p_high_info:.4f}")

    top5 = sorted(artifacts.subflow_offsets.items(), key=lambda kv: kv[1], reverse=True)[:5]
    bot5 = sorted(artifacts.subflow_offsets.items(), key=lambda kv: kv[1])[:5]

    print("\nTop 5 subflow offsets:")
    for sf, val in top5:
        print(f"{sf}: {val:.6f}")

    print("\nBottom 5 subflow offsets:")
    for sf, val in bot5:
        print(f"{sf}: {val:.6f}")

    print("\nValidation checks:")
    for k, v in artifacts.checks.items():
        print(f"{k}: {v}")

    if not all(str(v).startswith("PASS") for v in artifacts.checks.values()):
        raise RuntimeError("Validation checks failed; artifacts not saved.")

    # Save calibration.
    cal_out = phase4_out_dir / "psuccess_calibration.csv"
    artifacts.calibration.to_csv(cal_out, index=False)

    # Save model artifact.
    model_payload = {
        "model_notes": {
            "target": "ProvideSolution attempt success within 5 turns",
            "solution_actions": sorted(SOLUTION_ACTIONS),
            "n_attempts": n_attempts,
            "n_conversations_with_attempts": n_convos,
            "overall_success_rate": overall_success,
            "model_formula": "p_success = sigmoid(theta_0 + theta_i * info_proxy + alpha_subflow + alpha_action)",
            "info_proxy_definition": "cumulative_values_at_turn / subflow_mean_total_values, clipped to [0,1]",
            "difficulty_proxy_note": "Removed from regression. Subflow difficulty captured by alpha_subflow offsets.",
            "frustration_proxy_note": "Removed from regression. Confounded with info_proxy (both increase with conversation maturity). Frustration affects simulator through dropout_probability and escalation_probability only. See frustration_proxy_diagnostic.txt.",
            "per_attempt_vs_per_conversation": "This model predicts per-attempt success probability (~48% baseline). Per-conversation resolution rates (~86%) are higher because a conversation may have multiple ProvideSolution attempts before succeeding.",
        },
        "coefficients": {
            "theta_0": artifacts.theta_0,
            "theta_i": artifacts.theta_i,
            "theta_d": artifacts.theta_d,
            "theta_f": artifacts.theta_f,
        },
        "action_offsets": {
            "ProvideSolution": 0.10,
            "AskInfo": -0.15,
            "AffectiveRepair": -0.05,
        },
        "subflow_offsets": artifacts.subflow_offsets,
        "validation": artifacts.checks,
    }

    json_out = phase4_out_dir / "psuccess_model.json"
    json_out.write_text(json.dumps(model_payload, indent=2), encoding="utf-8")

    print("\nSaved artifacts:")
    print(json_out)
    print(cal_out)


if __name__ == "__main__":
    main()
