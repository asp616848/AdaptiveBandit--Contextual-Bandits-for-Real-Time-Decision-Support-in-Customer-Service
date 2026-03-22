from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from simulation_core.config import (
    ACTION_SPACE_8,
    ARTIFACTS_ROOT,
    DEFAULT_TIER_DISTRIBUTION,
    DOMAIN_OPENASSISTANT,
    DOMAIN_TWITTER,
    LOW_CONFIDENCE_THRESHOLD,
    OPENASSISTANT_TRAIN_PATH,
    OPENASSISTANT_VAL_PATH,
    REPORTS_ROOT,
    TIER_PRIOR_BY_DOMAIN,
    TIER_VALUE_WEIGHT,
    TIER_VALUES,
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


def load_twitter_threads(
    path: Path = TWITTER_PATH,
    max_threads: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    req = {"tweet_id", "in_response_to_tweet_id", "inbound", "text"}
    missing = req.difference(df.columns)
    if missing:
        raise ValueError(f"Twitter schema missing required columns: {sorted(missing)}")

    df = df.copy()
    df["text"] = df["text"].map(_clean_text)
    df = df[df["text"].str.len() > 0]

    # Keep only thread ids with >=2 rows to reduce memory and grouping overhead.
    valid_ids = (
        df.loc[df["in_response_to_tweet_id"].notna(), "in_response_to_tweet_id"]
        .value_counts()
        .loc[lambda s: s >= 2]
        .index
    )

    if max_threads is not None and max_threads > 0 and len(valid_ids) > max_threads:
        valid_ids = pd.Series(valid_ids).sample(n=max_threads, random_state=random_state).tolist()

    if len(valid_ids) == 0:
        return pd.DataFrame()

    df = df[df["in_response_to_tweet_id"].isin(valid_ids)]
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


def load_openassistant_turns(
    train_path: Path = OPENASSISTANT_TRAIN_PATH,
    val_path: Path = OPENASSISTANT_VAL_PATH,
    max_conversations: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
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

    conv_ids = df[conv_col].dropna().unique()
    if max_conversations is not None and max_conversations > 0 and len(conv_ids) > max_conversations:
        conv_ids = pd.Series(conv_ids).sample(n=max_conversations, random_state=random_state).tolist()
        df = df[df[conv_col].isin(conv_ids)]

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


def annotate_turns(
    turns_df: pd.DataFrame,
    max_workers: int | None = None,
    progress_every: int = 500,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 1000,
    resume: bool = True,
) -> pd.DataFrame:
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

    worker_count = max_workers if max_workers is not None else max(1, min(16, (os.cpu_count() or 4)))

    # Resume from an existing partial checkpoint when keys align.
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        try:
            ckpt = pd.read_csv(checkpoint_path, low_memory=False)
            key_cols = ["conv_id", "turn_id", "speaker_role", "text"]
            if all(c in ckpt.columns for c in key_cols) and all(c in out.columns for c in key_cols):
                annot_cols = [
                    "action_label",
                    "action_probs",
                    "action_confidence",
                    "sentiment_score",
                    "sentiment_confidence",
                    "frustration_score",
                    "frustration_confidence",
                    "annotator_confidence",
                ]
                existing_annot = [c for c in annot_cols if c in ckpt.columns]
                merged = out.merge(
                    ckpt[key_cols + existing_annot],
                    on=key_cols,
                    how="left",
                    suffixes=("", "_ckpt"),
                )
                for c in existing_annot:
                    ck = f"{c}_ckpt"
                    merged[c] = merged[c].where(merged[c].notna(), merged[ck])
                    merged.drop(columns=[ck], inplace=True)
                out = merged
                n_resumed = int(out["annotator_confidence"].notna().sum())
                print(f"[annotate_turns] Resumed {n_resumed}/{len(out)} rows from checkpoint: {checkpoint_path}")
            else:
                print("[annotate_turns] Checkpoint keys missing or incompatible; starting fresh.")
        except Exception as exc:
            print(f"[annotate_turns] Failed to load checkpoint ({checkpoint_path}): {exc}")

    pending_mask = out["annotator_confidence"].isna()
    agent_idx = out.index[(out["speaker_role"] == "agent") & pending_mask].tolist()
    customer_idx = out.index[(out["speaker_role"] == "customer") & pending_mask].tolist()

    def _annotate_agent(idx: int):
        text = out.at[idx, "text"]
        labeled = label_action_with_llm(text)
        return idx, labeled

    def _annotate_customer(idx: int):
        text = out.at[idx, "text"]
        scored = score_customer_state_with_llm(text)
        return idx, scored

    done = int(out["annotator_confidence"].notna().sum())
    total = len(out)

    if progress_every > 0 and done > 0:
        print(f"[annotate_turns] Starting from resumed progress {done}/{total}")

    if done >= total:
        print("[annotate_turns] All rows already annotated; skipping compute.")
        return out

    with ThreadPoolExecutor(max_workers=worker_count) as ex:
        futures = {ex.submit(_annotate_agent, idx): ("agent", idx) for idx in agent_idx}
        futures.update({ex.submit(_annotate_customer, idx): ("customer", idx) for idx in customer_idx})

        for fut in as_completed(futures):
            role, idx = futures[fut]
            result_idx, payload = fut.result()
            if role == "agent":
                out.at[result_idx, "action_label"] = payload["action_label"]
                out.at[result_idx, "action_probs"] = payload["action_probs"]
                out.at[result_idx, "action_confidence"] = payload["action_confidence"]
                out.at[result_idx, "annotator_confidence"] = payload["annotator_confidence"]
                out.at[result_idx, "sentiment_score"] = None
                out.at[result_idx, "sentiment_confidence"] = None
                out.at[result_idx, "frustration_score"] = None
                out.at[result_idx, "frustration_confidence"] = None
            else:
                out.at[result_idx, "action_label"] = None
                out.at[result_idx, "action_probs"] = None
                out.at[result_idx, "action_confidence"] = None
                out.at[result_idx, "sentiment_score"] = payload["sentiment_score"]
                out.at[result_idx, "sentiment_confidence"] = payload["sentiment_confidence"]
                out.at[result_idx, "frustration_score"] = payload["frustration_score"]
                out.at[result_idx, "frustration_confidence"] = payload["frustration_confidence"]
                out.at[result_idx, "annotator_confidence"] = payload["annotator_confidence"]

            done += 1
            if progress_every > 0 and (done % progress_every == 0 or done == total):
                print(f"[annotate_turns] {done}/{total} rows labeled")

            if checkpoint_path is not None and checkpoint_every > 0 and (done % checkpoint_every == 0 or done == total):
                tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
                out.to_csv(tmp, index=False)
                os.replace(tmp, checkpoint_path)
                print(f"[annotate_turns] checkpoint saved: {checkpoint_path} ({done}/{total})")

    return out


def _normalize_prob_dict(raw: Dict[str, float]) -> Dict[str, float]:
    vals = {k: float(max(raw.get(k, 0.0), 0.0)) for k in TIER_VALUES}
    total = sum(vals.values())
    if total <= 0.0:
        return {k: float(v) for k, v in zip(TIER_VALUES, DEFAULT_TIER_DISTRIBUTION)}
    return {k: v / total for k, v in vals.items()}


def _tier_signal_scores(text_blob: str, domain_source: str, conv_length: int) -> Dict[str, float]:
    base = dict(TIER_PRIOR_BY_DOMAIN.get(domain_source, {}))
    if not base:
        base = {k: v for k, v in zip(TIER_VALUES, DEFAULT_TIER_DISTRIBUTION)}

    # Start with log-space-friendly positive mass.
    scores = {k: float(max(base.get(k, 0.0), 1e-6)) for k in TIER_VALUES}
    lowered = (text_blob or "").lower()

    enterprise_keys = ["sla", "contract", "security", "compliance", "sso", "enterprise", "account manager"]
    business_keys = ["team", "workspace", "invoice", "billing cycle", "seat", "api", "integration"]
    pro_keys = ["subscription", "upgrade", "pro", "renew", "feature"]
    free_keys = ["free", "trial", "basic", "student", "can't pay"]

    for k in enterprise_keys:
        if k in lowered:
            scores["Enterprise"] += 0.08
            scores["Business+"] += 0.03
    for k in business_keys:
        if k in lowered:
            scores["Business+"] += 0.07
            scores["Pro"] += 0.02
    for k in pro_keys:
        if k in lowered:
            scores["Pro"] += 0.06
    for k in free_keys:
        if k in lowered:
            scores["Free"] += 0.07

    # Longer support threads are more likely from paying tiers.
    if conv_length >= 12:
        scores["Enterprise"] += 0.04
        scores["Business+"] += 0.03
    elif conv_length <= 4:
        scores["Free"] += 0.03

    return _normalize_prob_dict(scores)


def assign_probabilistic_tiers(turns_df: pd.DataFrame) -> pd.DataFrame:
    out = turns_df.copy()

    for field in ["tier", "tier_probs", "tier_confidence", "customer_value_weight"]:
        if field not in out.columns:
            out[field] = None

    for conv_id, g in out.groupby("conv_id", sort=False):
        domain = str(g["domain_source"].iloc[0])
        conv_length = int(g["conv_length"].iloc[0])
        text_blob = " ".join(g["text"].astype(str).tolist())

        probs = _tier_signal_scores(text_blob=text_blob, domain_source=domain, conv_length=conv_length)
        tier = max(probs, key=probs.get)
        conf = float(probs[tier])
        value_weight = float(TIER_VALUE_WEIGHT.get(tier, 0.4))

        idx = g.index
        out.loc[idx, "tier"] = tier
        out.loc[idx, "tier_probs"] = [probs] * len(idx)
        out.loc[idx, "tier_confidence"] = conf
        out.loc[idx, "customer_value_weight"] = value_weight

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
        "",
        "## Tier Distribution - Twitter",
        twitter_df["tier"].fillna("Unknown").value_counts(dropna=False).rename_axis("tier").reset_index(name="count").to_markdown(index=False),
        "",
        "## Tier Distribution - OpenAssistant",
        open_df["tier"].fillna("Unknown").value_counts(dropna=False).rename_axis("tier").reset_index(name="count").to_markdown(index=False),
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(
    output_dir: Path | None = None,
    twitter_max_threads: int | None = None,
    openassistant_max_conversations: int | None = None,
    random_state: int = 42,
    max_workers: int | None = None,
    progress_every: int = 500,
    resume_if_exists: bool = True,
    checkpoint_every: int = 1000,
) -> Dict[str, Path]:
    output_dir = ensure_dir(output_dir or (ARTIFACTS_ROOT / "revised_pipeline" / "labeled_data"))
    ensure_dir(REPORTS_ROOT)

    twitter_path = output_dir / "twitter_labeled.csv"
    open_path = output_dir / "openassistant_labeled.csv"
    index_path = output_dir / "turn_index.csv"
    report_path = REPORTS_ROOT / "labeling_report.md"

    if resume_if_exists and twitter_path.exists() and open_path.exists() and index_path.exists() and report_path.exists():
        print("[run_pipeline] Reusing existing labeled artifacts.")
        return {
            "twitter_labeled": twitter_path,
            "openassistant_labeled": open_path,
            "turn_index": index_path,
            "labeling_report": report_path,
        }

    print("[run_pipeline] Loading source turns...")

    twitter_turns = load_twitter_threads(max_threads=twitter_max_threads, random_state=random_state)
    open_turns = load_openassistant_turns(max_conversations=openassistant_max_conversations, random_state=random_state)

    twitter_turns = assign_probabilistic_tiers(twitter_turns)
    open_turns = assign_probabilistic_tiers(open_turns)

    print(f"[run_pipeline] Twitter turns: {len(twitter_turns)}")
    print(f"[run_pipeline] OpenAssistant turns: {len(open_turns)}")
    print("[run_pipeline] Annotating turns...")

    twitter_partial = output_dir / "twitter_labeled.partial.csv"
    open_partial = output_dir / "openassistant_labeled.partial.csv"

    twitter_labeled = annotate_turns(
        twitter_turns,
        max_workers=max_workers,
        progress_every=progress_every,
        checkpoint_path=twitter_partial,
        checkpoint_every=checkpoint_every,
        resume=resume_if_exists,
    )
    open_labeled = annotate_turns(
        open_turns,
        max_workers=max_workers,
        progress_every=progress_every,
        checkpoint_path=open_partial,
        checkpoint_every=checkpoint_every,
        resume=resume_if_exists,
    )

    twitter_labeled.to_csv(twitter_path, index=False)
    open_labeled.to_csv(open_path, index=False)

    print(f"[run_pipeline] Wrote: {twitter_path}")
    print(f"[run_pipeline] Wrote: {open_path}")

    index_df = pd.DataFrame(
        [
            {"domain_source": DOMAIN_TWITTER, "file_path": str(twitter_path), "n_rows": len(twitter_labeled)},
            {"domain_source": DOMAIN_OPENASSISTANT, "file_path": str(open_path), "n_rows": len(open_labeled)},
        ]
    )
    index_df.to_csv(index_path, index=False)

    write_labeling_report(twitter_labeled, open_labeled, report_path)
    print(f"[run_pipeline] Wrote report: {report_path}")

    # Cleanup partial checkpoints once final outputs are safely persisted.
    for p in [twitter_partial, open_partial]:
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

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
