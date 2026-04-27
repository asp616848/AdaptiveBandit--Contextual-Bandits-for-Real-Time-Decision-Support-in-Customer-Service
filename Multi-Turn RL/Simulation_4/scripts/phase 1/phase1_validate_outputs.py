from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def render_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<empty>"
    return df.to_string(index=False)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    base = root / "Simulation_4" / "artifacts" / "phase 1"

    files = {
        "extract1_conversation_stats.csv": base / "extract1_conversation_stats.csv",
        "extract2_action_sequences.csv": base / "extract2_action_sequences.csv",
        "extract3_subflow_stats.csv": base / "extract3_subflow_stats.csv",
        "extract4_information_by_turn.csv": base / "extract4_information_by_turn.csv",
        "extract5_behavioral_feature_table.csv": base / "extract5_behavioral_feature_table.csv",
        "abcd_phase1_overview.csv": base / "abcd_phase1_overview.csv",
    }

    out_lines: list[str] = []
    out_lines.append("ROW_COUNTS")
    for name, path in files.items():
        df = pd.read_csv(path)
        out_lines.append(f"{name}: {len(df)}")

    overview = pd.read_csv(files["abcd_phase1_overview.csv"])
    out_lines.append("\nOVERVIEW_FULL")
    out_lines.append(render_table(overview))

    convo = pd.read_csv(files["extract1_conversation_stats.csv"])
    out_lines.append("\nFULL_CORPUS_RATES")
    out_lines.append(f"resolution_rate={convo['resolution_flag'].mean():.6f}")
    out_lines.append(f"escalation_rate={convo['escalation_flag'].mean():.6f}")

    sub = pd.read_csv(files["extract3_subflow_stats.csv"])
    top_escalation = sub.sort_values(["escalation_rate", "conversation_count"], ascending=[False, False]).head(10)
    top_mean_turns = sub.sort_values(["mean_turns", "conversation_count"], ascending=[False, False]).head(10)

    out_lines.append("\nTOP_10_SUBFLOWS_BY_ESCALATION_RATE")
    out_lines.append(render_table(top_escalation))

    out_lines.append("\nTOP_10_SUBFLOWS_BY_MEAN_TURNS")
    out_lines.append(render_table(top_mean_turns))

    raw = json.loads((root / "abcd_v1.1.json").read_text(encoding="utf-8"))
    rows = []
    for split in ["train", "dev", "test"]:
        for convo_obj in raw.get(split, []):
            convo_id = convo_obj.get("convo_id")
            original = convo_obj.get("original") if isinstance(convo_obj.get("original"), list) else []
            delexed = convo_obj.get("delexed") if isinstance(convo_obj.get("delexed"), list) else []
            n = max(len(original), len(delexed))
            for i in range(n):
                o = original[i] if i < len(original) else None
                d = delexed[i] if i < len(delexed) else None
                speaker = o[0] if isinstance(o, list) and len(o) > 0 else (d.get("speaker") if isinstance(d, dict) else "")
                text = o[1] if isinstance(o, list) and len(o) > 1 else (d.get("text") if isinstance(d, dict) else "")
                targets = d.get("targets") if isinstance(d, dict) else []
                action = targets[2] if isinstance(targets, list) and len(targets) > 2 else ""
                turn_count = d.get("turn_count") if isinstance(d, dict) else None
                rows.append(
                    {
                        "convo_id": convo_id,
                        "turn_count": int(turn_count) if turn_count is not None else i + 1,
                        "speaker": str(speaker).lower(),
                        "text": str(text),
                        "action": str(action),
                    }
                )

    turns = pd.DataFrame(rows).sort_values(["convo_id", "turn_count"], kind="mergesort")
    actions = pd.read_csv(files["extract2_action_sequences.csv"])
    escalation_ids = actions[actions["has_notify_team"] == 1]["convo_id"].tolist()[:3]

    out_lines.append("\nESCALATION_SAMPLES")
    for cid in escalation_ids:
        out_lines.append(f"--- convo_id={cid} ---")
        c = turns[turns["convo_id"] == cid]
        for _, row in c.iterrows():
            extra = f" | action={row['action']}" if row["speaker"] == "action" else ""
            out_lines.append(f"t{int(row['turn_count']):02d} [{row['speaker']}] {row['text']}{extra}")

    report_path = base / "phase1_validation_report.txt"
    report_text = "\n".join(out_lines) + "\n"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    main()
