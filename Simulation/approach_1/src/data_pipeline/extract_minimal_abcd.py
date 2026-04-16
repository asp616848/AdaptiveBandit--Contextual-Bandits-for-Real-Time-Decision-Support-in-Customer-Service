from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "nan", "-1"}:
        return None
    return text.replace(" ", "_")


def _iter_conversations(payload: Any):
    if isinstance(payload, dict):
        for split_name in ("train", "dev", "test"):
            split = payload.get(split_name, [])
            if isinstance(split, list):
                for convo in split:
                    yield split_name, convo
        return

    if isinstance(payload, list):
        for convo in payload:
            yield "unknown", convo


def _extract_action_sequence(delexed_turns: list[dict[str, Any]]) -> list[str]:
    sequence: list[str] = []

    for turn in delexed_turns:
        if not isinstance(turn, dict):
            continue

        targets = turn.get("targets", [])
        action = None
        if isinstance(targets, list) and len(targets) >= 3:
            action = _normalize_label(targets[2])

        if action is None:
            continue

        if not sequence or sequence[-1] != action:
            sequence.append(action)

    return sequence


def _extract_success_label(delexed_turns: list[dict[str, Any]]) -> int:
    for turn in delexed_turns:
        if not isinstance(turn, dict):
            continue
        targets = turn.get("targets", [])
        if isinstance(targets, list) and len(targets) >= 2:
            nextstep = _normalize_label(targets[1])
            if nextstep == "end_conversation":
                return 1
    return 0


def _build_catalog(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_subflow: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_subflow[r["subflow"]].append(r)

    catalog: list[dict[str, Any]] = []
    for subflow, rows in by_subflow.items():
        success_rows = [r for r in rows if r["success"] == 1 and r["actions_required"]]
        pool = success_rows if success_rows else [r for r in rows if r["actions_required"]]
        if not pool:
            continue

        seq_counts = Counter(tuple(r["actions_required"]) for r in pool)
        canonical_seq = list(seq_counts.most_common(1)[0][0])
        success_rate = sum(r["success"] for r in rows) / max(len(rows), 1)

        catalog.append(
            {
                "subflow": subflow,
                "canonical_actions": canonical_seq,
                "support_count": len(pool),
                "empirical_success_rate": round(float(success_rate), 6),
            }
        )

    catalog.sort(key=lambda x: (x["subflow"]))
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract minimal ABCD tasks for Approach 1 simulator.")
    parser.add_argument("--input", required=True, help="Path to abcd_v1.1.json")
    parser.add_argument("--out-dir", default="Simulation_4/approach_1/data/processed", help="Output directory")
    parser.add_argument("--min-actions", type=int, default=1, help="Skip conversations with fewer than this many actions")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))

    records: list[dict[str, Any]] = []
    for split, convo in _iter_conversations(payload):
        if not isinstance(convo, dict):
            continue

        convo_id = str(convo.get("convo_id", "unknown_convo"))
        scenario = convo.get("scenario", {}) if isinstance(convo.get("scenario"), dict) else {}
        subflow = _normalize_label(scenario.get("subflow")) or "unknown_subflow"

        delexed = convo.get("delexed", [])
        if not isinstance(delexed, list):
            continue

        actions_required = _extract_action_sequence(delexed)
        if len(actions_required) < args.min_actions:
            continue

        success = _extract_success_label(delexed)

        records.append(
            {
                "convo_id": convo_id,
                "split": split,
                "subflow": subflow,
                "actions_required": actions_required,
                "success": success,
            }
        )

    catalog = _build_catalog(records)

    tasks_path = out_dir / "tasks_minimal.jsonl"
    with tasks_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")

    catalog_path = out_dir / "subflow_catalog.json"
    catalog_path.write_text(json.dumps({"subflows": catalog}, indent=2), encoding="utf-8")

    seq_lengths = [len(r["actions_required"]) for r in records]
    summary = {
        "num_conversations": len(records),
        "num_subflows": len({r["subflow"] for r in records}),
        "sequence_len_min": min(seq_lengths) if seq_lengths else 0,
        "sequence_len_mean": (sum(seq_lengths) / len(seq_lengths)) if seq_lengths else 0.0,
        "sequence_len_max": max(seq_lengths) if seq_lengths else 0,
        "tasks_path": str(tasks_path),
        "catalog_path": str(catalog_path),
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
