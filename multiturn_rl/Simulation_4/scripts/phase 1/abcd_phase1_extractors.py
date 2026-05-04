from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


TARGETS_SCHEMA = ["intent", "nextstep", "action", "value", "utterance_ranking"]

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

STILL_THERE_RE = re.compile(
    r"\b(?:are you still there|still there\?|are you there|hello\?|"
    r"this chat will close|remain inactive)\b",
    re.IGNORECASE,
)


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


def load_corpus(root: Path) -> list[dict[str, Any]]:
    src = root / "abcd_v1.1.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    merged: list[dict[str, Any]] = []
    for split in ["train", "dev", "test"]:
        merged.extend(data.get(split, []))
    return merged


def build_turn_table(corpus: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for convo in corpus:
        convo_id = convo.get("convo_id")
        scenario = convo.get("scenario") if isinstance(convo.get("scenario"), dict) else {}
        personal = scenario.get("personal") if isinstance(scenario.get("personal"), dict) else {}

        member_level = personal.get("member_level", "<NA>")
        flow = scenario.get("flow", "<NA>")
        subflow = scenario.get("subflow", "<NA>")

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
                    "member_level": as_str(member_level) or "<NA>",
                    "flow": as_str(flow) or "<NA>",
                    "subflow": as_str(subflow) or "<NA>",
                    "turn_count": int(turn_count),
                    "speaker": as_str(speaker).lower(),
                    "text": as_str(text),
                    "intent": parsed["intent"],
                    "nextstep": parsed["nextstep"],
                    "action": parsed["action"],
                    "value": parsed["value"],
                    "utterance_ranking": parsed["utterance_ranking"],
                }
            )

    turns = pd.DataFrame(rows)
    turns = turns.sort_values(["convo_id", "turn_count"], kind="mergesort").reset_index(drop=True)
    turns["value_count"] = turns["value"].apply(lambda v: len(v) if isinstance(v, list) else 0)
    turns["is_closing_turn"] = (
        turns["speaker"].eq("agent")
        & turns["text"].fillna("").astype(str).str.contains(CLOSING_RE, na=False)
    )
    turns["is_customer_requested_escalation"] = (
        turns["speaker"].eq("customer")
        & turns["text"].fillna("").astype(str).str.contains(CUSTOMER_ESCALATION_RE, na=False)
    )
    turns["is_still_there_turn"] = (
        turns["speaker"].eq("agent")
        & turns["text"].fillna("").astype(str).str.contains(STILL_THERE_RE, na=False)
    )
    turns["is_escalation_action"] = (
        turns["speaker"].eq("action")
        & turns["action"].astype(str).str.lower().eq("notify-team")
    )
    return turns


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    out_dir = root / "Simulation_4" / "artifacts" / "phase 1"
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(root)
    turns = build_turn_table(corpus)

    # Extract 1: conversation-level statistics
    convo_stats = (
        turns.groupby("convo_id", as_index=False)
        .agg(
            member_level=("member_level", "first"),
            flow=("flow", "first"),
            subflow=("subflow", "first"),
            total_turns=("turn_count", "count"),
            customer_turns=("speaker", lambda s: int((s == "customer").sum())),
            agent_turns=("speaker", lambda s: int((s == "agent").sum())),
            action_turns=("speaker", lambda s: int((s == "action").sum())),
            _has_closing_turn=("is_closing_turn", lambda s: int(bool(s.any()))),
            _has_customer_requested_escalation=("is_customer_requested_escalation", lambda s: int(bool(s.any()))),
            _has_action_escalation=("is_escalation_action", lambda s: int(bool(s.any()))),
        )
    )

    convo_stats["escalation_flag"] = (
        (convo_stats["_has_action_escalation"] == 1)
        | (convo_stats["_has_customer_requested_escalation"] == 1)
    ).astype(int)

    def signal1_dropout(group: pd.DataFrame) -> int:
        still_turns = group.loc[group["is_still_there_turn"], "turn_count"].tolist()
        if not still_turns:
            return 0
        customer_turns = group.loc[group["speaker"] == "customer", "turn_count"].tolist()
        for st in still_turns:
            if not any(ct > st for ct in customer_turns):
                return 1
        return 0

    dropout_signal1 = (
        turns.groupby("convo_id", sort=False)
        .apply(signal1_dropout)
        .rename("dropout_flag")
        .reset_index()
    )
    convo_stats = convo_stats.merge(dropout_signal1, on="convo_id", how="left")
    convo_stats["dropout_flag"] = convo_stats["dropout_flag"].fillna(0).astype(int)

    # Resolved means a natural closing happened and there was no escalation.
    convo_stats["resolution_flag"] = (
        (convo_stats["_has_closing_turn"] == 1)
        & (convo_stats["escalation_flag"] == 0)
    ).astype(int)

    # Enforce strict priority: escalation > dropout > resolution.
    convo_stats.loc[convo_stats["escalation_flag"] == 1, ["dropout_flag", "resolution_flag"]] = 0
    convo_stats.loc[
        (convo_stats["escalation_flag"] == 0) & (convo_stats["dropout_flag"] == 1),
        "resolution_flag",
    ] = 0

    convo_stats = convo_stats.drop(columns=["_has_closing_turn", "_has_customer_requested_escalation", "_has_action_escalation"])

    # Extract 2: action sequence per conversation
    action_turns = turns[turns["speaker"] == "action"].copy()
    action_seq = (
        action_turns.groupby("convo_id", as_index=False)
        .agg(
            action_sequence=("action", lambda s: "|".join([as_str(x) for x in s if as_str(x)])),
            action_count=("action", "size"),
            unique_actions=("action", lambda s: "|".join(sorted(set([as_str(x) for x in s if as_str(x)])))),
            has_verify_identity=("action", lambda s: int(any(as_str(x).lower() == "verify-identity" for x in s))),
            has_pull_up_account=("action", lambda s: int(any(as_str(x).lower() == "pull-up-account" for x in s))),
            has_offer_refund=("action", lambda s: int(any(as_str(x).lower() == "offer-refund" for x in s))),
            has_notify_team=("action", lambda s: int(any(as_str(x).lower() == "notify-team" for x in s))),
        )
    )

    action_seq = convo_stats[["convo_id"]].merge(action_seq, on="convo_id", how="left")
    for c in ["action_sequence", "unique_actions"]:
        action_seq[c] = action_seq[c].fillna("")
    for c in [
        "action_count",
        "has_verify_identity",
        "has_pull_up_account",
        "has_offer_refund",
        "has_notify_team",
    ]:
        action_seq[c] = action_seq[c].fillna(0).astype(int)

    # Extract 3: subflow-level statistics
    subflow_stats = (
        convo_stats.groupby("subflow", as_index=False)
        .agg(
            mean_turns=("total_turns", "mean"),
            mean_action_count=("action_turns", "mean"),
            resolution_rate=("resolution_flag", "mean"),
            escalation_rate=("escalation_flag", "mean"),
            conversation_count=("convo_id", "size"),
        )
        .sort_values(["conversation_count", "subflow"], ascending=[False, True])
        .reset_index(drop=True)
    )

    # Difficulty proxy normalized by expected action effort for the same subflow.
    subflow_action_mean = subflow_stats.set_index("subflow")["mean_action_count"].to_dict()
    convo_stats["subflow_mean_action_count"] = convo_stats["subflow"].map(subflow_action_mean)
    convo_stats["subflow_mean_action_count"] = convo_stats["subflow_mean_action_count"].fillna(1.0)
    convo_stats["subflow_mean_action_count"] = convo_stats["subflow_mean_action_count"].replace(0, 1.0)
    convo_stats["difficulty_proxy"] = convo_stats["action_turns"] / convo_stats["subflow_mean_action_count"]
    convo_stats = convo_stats.drop(columns=["subflow_mean_action_count"])

    # Extract 4: information provision per action turn.
    action_turns = turns[turns["speaker"] == "action"].copy()
    action_turns = action_turns.sort_values(["convo_id", "turn_count"], kind="mergesort")
    action_turns["cumulative_values_by_turn"] = action_turns.groupby("convo_id")["value_count"].cumsum()
    info_per_turn = action_turns[["convo_id", "turn_count", "value_count", "cumulative_values_by_turn"]].copy()

    # Extract 5: behavioral feature table
    subflow_mean_turns = subflow_stats.set_index("subflow")["mean_turns"].to_dict()

    value_by_convo = action_turns.groupby("convo_id", as_index=False).agg(
        total_values_provided=("value_count", "sum"),
    )

    progress_by_convo = turns.groupby("convo_id", as_index=False).agg(
        progress_turns=(
            "nextstep",
            lambda s: int(
                pd.Series(s)
                .astype(str)
                .str.lower()
                .isin(["take_action", "end_conversation"])
                .sum()
            ),
        ),
        total_turns=("turn_count", "size"),
    )
    progress_by_convo["forward_progress_rate"] = progress_by_convo["progress_turns"] / progress_by_convo["total_turns"]

    behavioral = convo_stats[[
        "convo_id",
        "member_level",
        "subflow",
        "total_turns",
        "escalation_flag",
        "resolution_flag",
        "action_turns",
    ]].merge(value_by_convo, on="convo_id", how="left")

    behavioral = behavioral.merge(progress_by_convo[["convo_id", "forward_progress_rate"]], on="convo_id", how="left")
    behavioral["total_values_provided"] = behavioral["total_values_provided"].fillna(0)

    behavioral["subflow_mean_turns"] = behavioral["subflow"].map(subflow_mean_turns).fillna(behavioral["total_turns"].mean())
    behavioral["relative_turn_count"] = behavioral["total_turns"] / behavioral["subflow_mean_turns"].replace(0, 1)
    behavioral["avg_values_per_action_turn"] = behavioral["total_values_provided"] / behavioral["action_turns"].replace(0, 1)
    behavioral["action_count"] = behavioral["action_turns"]

    behavioral = behavioral[[
        "convo_id",
        "member_level",
        "subflow",
        "relative_turn_count",
        "total_values_provided",
        "avg_values_per_action_turn",
        "escalation_flag",
        "resolution_flag",
        "action_count",
        "forward_progress_rate",
    ]].copy()

    # Overview table
    overview = (
        convo_stats.groupby("member_level", as_index=False)
        .agg(
            conversations=("convo_id", "size"),
            mean_turns=("total_turns", "mean"),
            p95_turns=("total_turns", lambda s: float(pd.Series(s).quantile(0.95))),
            resolution_rate=("resolution_flag", "mean"),
            escalation_rate=("escalation_flag", "mean"),
            dropout_rate=("dropout_flag", "mean"),
            mean_action_count=("action_turns", "mean"),
        )
        .sort_values("conversations", ascending=False)
        .reset_index(drop=True)
    )

    # Save outputs
    convo_stats.to_csv(out_dir / "extract1_conversation_stats.csv", index=False)
    action_seq.to_csv(out_dir / "extract2_action_sequences.csv", index=False)
    subflow_stats.to_csv(out_dir / "extract3_subflow_stats.csv", index=False)
    info_per_turn.to_csv(out_dir / "extract4_information_by_turn.csv", index=False)
    behavioral.to_csv(out_dir / "extract5_behavioral_feature_table.csv", index=False)
    overview.to_csv(out_dir / "abcd_phase1_overview.csv", index=False)

    print("ABCD Phase 1 extraction complete")
    print(f"conversations={len(convo_stats)}")
    print(f"turns={len(turns)}")
    print(f"output_dir={out_dir}")


if __name__ == "__main__":
    main()
