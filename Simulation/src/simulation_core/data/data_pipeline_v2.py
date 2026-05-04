from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from simulation_core.config import (
    ACTION_SPACE_8,
    ARTIFACTS_ROOT,
    DOMAIN_OPENASSISTANT,
    DOMAIN_TWITTER,
    LOW_CONFIDENCE_THRESHOLD,
    OPENASSISTANT_TRAIN_PATH,
    OPENASSISTANT_VAL_PATH,
    REPORTS_ROOT,
    TWITTER_PATH,
)
from simulation_core.labeling.llm_labeler import label_action_with_llm, score_customer_state_with_llm
from simulation_core.utils.io_utils import ensure_dir


def _clean_text(text: object) -> str:
    if text is None:
        return ""
    val = str(text).strip()
    return val


def _speaker_from_twitter_inbound(v: object) -> str:
    # In TWCS, inbound=True means customer; False means agent reply.
    sval = str(v).strip().lower()
    return "customer" if sval in {"true", "1"} else "agent"


def load_twitter_threads(path: Path = TWITTER_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    req = {"tweet_id", "in_response_to_tweet_id", "inbound", "text"}
    missing = req.difference(df.columns)
    if missing:
        raise ValueError(f"Twitter schema missing required columns: {sorted(missing)}")

    df = df.copy()
    df["text"] = df["text"].map(_clean_text)
    df = df[df["text"].str.len() > 0]

    # Group by parent tweet id; keep conversational threads with >=2 turns.
    grouped = df.groupby("in_response_to_tweet_id", dropna=False)
    rows = []
    for parent, chunk in grouped:
        if pd.isna(parent):
            continue
        if len(chunk) < 2:
            continue

        if "created_at" in chunk.columns:
            chunk = chunk.sort_values("created_at")
        else:
            chunk = chunk.sort_values("tweet_id")

        conv_id = f"tw_{parent}"
        conv_length = len(chunk)
        for i, (_, r) in enumerate(chunk.iterrows(), start=1):
            rows.append(
                {
                    "conv_id": conv_id,
                    "turn_id": int(i),
                    "domain_source": DOMAIN_TWITTER,
                    "speaker_role": _speaker_from_twitter_inbound(r["inbound"]),
                    "text": _clean_text(r["text"]),
                    "text_length": len(_clean_text(r["text"])),
                    "turn_index": int(i),
                    "conv_length": int(conv_length),
                }
            )

    return pd.DataFrame(rows)


def load_openassistant_turns(train_path: Path = OPENASSISTANT_TRAIN_PATH, val_path: Path = OPENASSISTANT_VAL_PATH) -> pd.DataFrame:
    frames = []
    for p in [train_path, val_path]:
        if p.exists():
            frames.append(pd.read_csv(p, low_memory=False))

    if not frames:
        raise FileNotFoundError("OpenAssistant files were not found.")

    df = pd.concat(frames, ignore_index=True)
    colset = set(df.columns)

    text_col = "text" if "text" in colset else "message"
    role_col = "role" if "role" in colset else None

    if text_col is None:
        raise ValueError("OpenAssistant data must contain text/message column.")

    if "message_tree_id" in colset:
        conv_col = "message_tree_id"
    elif "conversation_id" in colset:
        conv_col = "conversation_id"
    elif "parent_id" in colset:
        conv_col = "parent_id"
    else:
        # Fallback to one row per synthetic conversation to preserve schema.
        df = df.copy()
        df["_fallback_conv_id"] = [f"oasst_{i}" for i in range(len(df))]
        conv_col = "_fallback_conv_id"

    df = df.copy()
    df[text_col] = df[text_col].map(_clean_text)
    df = df[df[text_col].str.len() > 0]

    rows: List[Dict[str, object]] = []
    for conv_id, chunk in df.groupby(conv_col, dropna=False):
        if pd.isna(conv_id):
            continue
        conv_length = len(chunk)
        if conv_length < 2:
            continue

        if "created_date" in chunk.columns:
            chunk = chunk.sort_values("created_date")
        elif "created_at" in chunk.columns:
            chunk = chunk.sort_values("created_at")

        for i, (_, r) in enumerate(chunk.iterrows(), start=1):
            raw_role = str(r.get(role_col, "assistant" if i % 2 == 0 else "user")).lower() if role_col else ("assistant" if i % 2 == 0 else "user")
            speaker = "agent" if raw_role in {"assistant", "agent", "support", "system"} else "customer"
            text = _clean_text(r[text_col])
            rows.append(
                {
                    "conv_id": f"oasst_{conv_id}",
                    "turn_id": int(i),
                    "domain_source": DOMAIN_OPENASSISTANT,
                    "speaker_role": speaker,
                    "text": text,
                    "text_length": len(text),
                    "turn_index": int(i),
                    "conv_length": int(conv_length),
                }
            )

    return pd.DataFrame(rows)


def annotate_turns(turns_df: pd.DataFrame) -> pd.DataFrame:
    out = turns_df.copy()

    for field in [
        "action_label",
        "action_probs",
        "action_confidence",
        "sentiment_score",
        "sentiment_confidence",
        "frustration_score",
        "frustration_confidence",
        "annotator_confidence",
    ]:
        if field not in out.columns:
            out[field] = None

    for idx, row in out.iterrows():
        text = row["text"]
        if row["speaker_role"] == "agent":
            labeled = label_action_with_llm(text)
            out.at[idx, "action_label"] = labeled["action_label"]
            out.at[idx, "action_probs"] = labeled["action_probs"]
            out.at[idx, "action_confidence"] = labeled["action_confidence"]
            out.at[idx, "annotator_confidence"] = labeled["annotator_confidence"]
            out.at[idx, "sentiment_score"] = None
            out.at[idx, "sentiment_confidence"] = None
            out.at[idx, "frustration_score"] = None
            out.at[idx, "frustration_confidence"] = None
        else:
            scored = score_customer_state_with_llm(text)
            out.at[idx, "action_label"] = None
            out.at[idx, "action_probs"] = None
            out.at[idx, "action_confidence"] = None
            out.at[idx, "sentiment_score"] = scored["sentiment_score"]
            out.at[idx, "sentiment_confidence"] = scored["sentiment_confidence"]
            out.at[idx, "frustration_score"] = scored["frustration_score"]
            out.at[idx, "frustration_confidence"] = scored["frustration_confidence"]
            out.at[idx, "annotator_confidence"] = scored["annotator_confidence"]

    return out


def _domain_report(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    agent = df[df["speaker_role"] == "agent"].copy()
    customer = df[df["speaker_role"] == "customer"].copy()

    label_dist = (
        agent["action_label"].fillna("Unknown").value_counts(dropna=False).rename_axis("action_label").reset_index(name="count")
    )

    conf_rows = [
        {
            "metric": "mean_action_confidence",
            "value": float(agent["action_confidence"].dropna().mean()) if not agent.empty else 0.0,
        },
        {
            "metric": "mean_sentiment_confidence",
            "value": float(customer["sentiment_confidence"].dropna().mean()) if not customer.empty else 0.0,
        },
        {
            "metric": "mean_frustration_confidence",
            "value": float(customer["frustration_confidence"].dropna().mean()) if not customer.empty else 0.0,
        },
    ]

    low_frac = float((df["annotator_confidence"].fillna(0.0) < LOW_CONFIDENCE_THRESHOLD).mean())
    return label_dist, pd.DataFrame(conf_rows), low_frac


def write_labeling_report(twitter_df: pd.DataFrame, open_df: pd.DataFrame, out_path: Path) -> None:
    tw_label, tw_conf, tw_low = _domain_report(twitter_df)
    oa_label, oa_conf, oa_low = _domain_report(open_df)

    tw_conv_len = twitter_df.groupby("conv_id")["conv_length"].max()
    oa_conv_len = open_df.groupby("conv_id")["conv_length"].max()

    lines = [
        "# Labeling Report (Revised Pipeline)",
        "",
        "## Action Label Distribution - Twitter",
        tw_label.to_markdown(index=False),
        "",
        "## Action Label Distribution - OpenAssistant",
        oa_label.to_markdown(index=False),
        "",
        "## Mean Confidence Metrics - Twitter",
        tw_conf.to_markdown(index=False),
        "",
        "## Mean Confidence Metrics - OpenAssistant",
        oa_conf.to_markdown(index=False),
        "",
        f"Low-confidence fraction (< {LOW_CONFIDENCE_THRESHOLD}) - Twitter: {tw_low:.4f}",
        f"Low-confidence fraction (< {LOW_CONFIDENCE_THRESHOLD}) - OpenAssistant: {oa_low:.4f}",
        "",
        "## Conversation Length Comparison",
        f"Twitter mean/median: {tw_conv_len.mean():.2f} / {tw_conv_len.median():.2f}",
        f"OpenAssistant mean/median: {oa_conv_len.mean():.2f} / {oa_conv_len.median():.2f}",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(output_dir: Path | None = None) -> Dict[str, Path]:
    output_dir = ensure_dir(output_dir or (ARTIFACTS_ROOT / "revised_pipeline" / "labeled_data"))
    ensure_dir(REPORTS_ROOT)

    twitter_turns = load_twitter_threads()
    open_turns = load_openassistant_turns()

    twitter_labeled = annotate_turns(twitter_turns)
    open_labeled = annotate_turns(open_turns)

    twitter_path = output_dir / "twitter_labeled.csv"
    open_path = output_dir / "openassistant_labeled.csv"
    index_path = output_dir / "turn_index.csv"

    twitter_labeled.to_csv(twitter_path, index=False)
    open_labeled.to_csv(open_path, index=False)

    index_df = pd.DataFrame(
        [
            {"domain_source": DOMAIN_TWITTER, "file_path": str(twitter_path), "n_rows": len(twitter_labeled)},
            {"domain_source": DOMAIN_OPENASSISTANT, "file_path": str(open_path), "n_rows": len(open_labeled)},
        ]
    )
    index_df.to_csv(index_path, index=False)

    report_path = REPORTS_ROOT / "labeling_report.md"
    write_labeling_report(twitter_labeled, open_labeled, report_path)

    return {
        "twitter_labeled": twitter_path,
        "openassistant_labeled": open_path,
        "turn_index": index_path,
        "labeling_report": report_path,
    }


if __name__ == "__main__":
    outputs = run_pipeline()
    for k, v in outputs.items():
        print(f"{k}: {v}")
