"""Task 1 data-driven calibration for hidden-state simulator parameters.

This script learns initial distributions and coefficients directly from the
Twitter Customer Support and OpenAssistant datasets available in DATA_PATHS.

Outputs:
  CustomerSupportBandit/outputs/task1_calibration.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import re

import numpy as np
import pandas as pd

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import DATA_PATHS, OUTPUT_DIR, TIER_NAMES
from data.feature_engineer import compute_sentiment
from data.openassistant_loader import build_conversation_trees, load_openassistant_data
from data.twitter_loader import add_escalation_labels, load_twitter_data, reconstruct_threads


PROBLEM_SIGNALS: Dict[str, List[Tuple[str, float]]] = {
    "billing": [
        (r"\bbill(ing)?\b", 1.2),
        (r"\bcharg(e|ed|ing)?\b", 1.1),
        (r"\bovercharg(ed|e)?\b|\bdouble\s+charg(e|ed)?\b", 1.4),
        (r"\binvoice\b", 1.2),
        (r"\bpayment\b|\bpay(ing|ment)?\b", 1.1),
        (r"\bcredit\s+card\b|\bdebit\s+card\b", 1.0),
        (r"\bsubscription\b", 1.0),
        (r"\brenew(al|ed)?\b", 0.8),
    ],
    "login": [
        (r"\blog\s?in\b|\bsign\s?in\b", 1.2),
        (r"\bpassword\b", 1.0),
        (r"\b2fa\b|\bmfa\b|\bauth(entication)?\b", 1.1),
        (r"\botp\b|\bverify\b|\bverification\b", 0.8),
        (r"\bcaptcha\b|\breset\s+password\b", 0.9),
    ],
    "bug": [
        (r"\bbug\b|\bissue\b|\bbroken\b|\bglitch\b", 1.1),
        (r"\berror\b|\bexception\b|\btraceback\b", 1.2),
        (r"\bcrash(ed|ing)?\b|\bfail(ed|ure)?\b", 1.1),
        (r"\bnot\s+work(ing)?\b|\bdoesn['’]t\s+work\b", 0.9),
    ],
    "refund": [
        (r"\brefund\b|\breimburse\b", 1.3),
        (r"\bmoney\s+back\b", 1.2),
        (r"\bcharged\s+twice\b", 1.3),
        (r"\breturn\b|\bchargeback\b", 0.9),
        (r"\bcancel(lation)?\b", 0.7),
    ],
    "integration": [
        (r"\bapi\b|\bendpoint\b|\btoken\b", 1.1),
        (r"\bwebhook\b|\bsdk\b|\bintegration\b", 1.3),
        (r"\boauth\b|\bcallback\b|\baccess\s+token\b", 1.0),
        (r"\bgraphql\b|\brest\b", 0.7),
    ],
    "performance": [
        (r"\bslow\b|\blag\b|\blatency\b", 1.2),
        (r"\btimeout\b|\boutage\b|\bdown\b", 1.2),
        (r"\bperformance\b|\bresponse\s+time\b", 1.0),
        (r"\bstuck\b|\bloading\b|\bspinning\b", 0.8),
        (r"\bunavailable\b|\bdegraded\b", 0.9),
    ],
    "account_access": [
        (r"\blocked\s+out\b|\blocked\b", 1.2),
        (r"\baccess\b|\bpermission\b", 1.0),
        (r"\baccount\b|\bdisabled\b|\bsuspended\b", 1.0),
        (r"\bcannot\s+access\b|\bcan't\s+access\b", 1.2),
        (r"\breactivat(e|ion)\b|\bunlock\b", 1.0),
        (r"\bnot\s+authorized\b|\bforbidden\b|\b403\b", 1.0),
    ],
    "inquiry": [
        (r"\bquestion\b|\bdoubt\b|\bcurious\b|\bwondering\b", 0.9),
        (r"\bhow\s+do\s+i\b|\bcan\s+i\b|\bis\s+there\s+a\s+way\b", 0.8),
        (r"\bwhat\s+is\b|\bwhere\s+can\s+i\b|\bany\s+tips\b", 0.7),
        (r"\blearn\s+more\b|\bbest\s+way\b|\bguidance\b", 0.7),
    ],
}

BASE_PROBLEM_TYPES: List[str] = [
    "billing",
    "login",
    "bug",
    "refund",
    "integration",
    "performance",
    "account_access",
    "inquiry",
    "other",
]

OTHER_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "you",
    "your",
    "have",
    "from",
    "just",
    "can",
    "cant",
    "not",
    "are",
    "was",
    "were",
    "our",
    "please",
    "help",
    "thanks",
    "thank",
    "been",
    "still",
    "into",
    "about",
    "what",
    "when",
    "where",
    "would",
    "should",
    "could",
    "there",
    "they",
    "them",
    "their",
    "need",
    "needs",
    "want",
    "also",
    "like",
    "http",
    "https",
    "com",
}


def _clip01(v: np.ndarray | float) -> np.ndarray | float:
    return np.clip(v, 0.0, 1.0)


def _safe_text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _normalize_category_name(value: str) -> str:
    s = re.sub(r"[^a-z0-9_ ]+", "", _safe_text(value).strip().lower())
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    raw = _safe_text(text).strip()
    if not raw:
        return None

    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def infer_problem_type_with_confidence(text: str) -> Tuple[str, float]:
    low = _safe_text(text).lower()
    if not low:
        return "other", 0.0

    scores: Dict[str, float] = {k: 0.0 for k in PROBLEM_SIGNALS.keys()}
    for label, rules in PROBLEM_SIGNALS.items():
        for pattern, weight in rules:
            matches = re.findall(pattern, low)
            if matches:
                scores[label] += weight * float(len(matches))

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    if top_score <= 0:
        return "other", 0.0

    # Confidence combines absolute support and margin over runner-up.
    margin = max(top_score - second_score, 0.0)
    conf = float(np.clip((0.55 * np.tanh(top_score / 2.5)) + (0.45 * np.tanh(margin / 2.0)), 0.0, 1.0))

    # Keep low-confidence assignments as "other" to reduce misclassification noise.
    if conf < 0.30:
        return "other", conf
    return top_label, conf


def _top_terms(texts: pd.Series, *, top_k: int, ngram_size: int = 1) -> List[Dict[str, object]]:
    counts: Counter[Tuple[str, ...]] = Counter()
    for text in texts:
        low = _safe_text(text).lower()
        tokens = re.findall(r"\b[a-z][a-z0-9_]{2,}\b", low)
        filtered = [tok for tok in tokens if tok not in OTHER_STOPWORDS]
        if len(filtered) < ngram_size:
            continue
        for i in range(len(filtered) - ngram_size + 1):
            grams = tuple(filtered[i : i + ngram_size])
            counts[grams] += 1

    top = counts.most_common(top_k)
    return [{"term": " ".join(k), "count": int(v)} for k, v in top]


def _build_problem_type_diagnostics(
    full_text: pd.Series,
    labels: pd.Series,
    confidence: pd.Series,
    *,
    top_k: int = 25,
) -> Dict[str, object]:
    total = int(labels.shape[0])
    other_mask = labels == "other"
    other_count = int(other_mask.sum())
    other_rate = float(other_count / max(total, 1))
    conf_arr = confidence.astype(float)

    other_texts = full_text[other_mask]
    sample_texts = [
        _safe_text(v)[:200]
        for v in other_texts.head(15).tolist()
        if _safe_text(v).strip()
    ]

    return {
        "total_rows": total,
        "other_count": other_count,
        "other_rate": round(other_rate, 6),
        "mean_confidence": round(float(conf_arr.mean()), 6),
        "p10_confidence": round(float(conf_arr.quantile(0.10)), 6),
        "p50_confidence": round(float(conf_arr.quantile(0.50)), 6),
        "top_other_unigrams": _top_terms(other_texts, top_k=top_k, ngram_size=1),
        "top_other_bigrams": _top_terms(other_texts, top_k=top_k, ngram_size=2),
        "other_examples": sample_texts,
    }


def _llm_reclassify_problem_types(
    full_text: pd.Series,
    weak_labels: pd.Series,
    weak_conf: pd.Series,
    *,
    enabled: bool,
    low_conf_threshold: float,
    max_calls: int,
    new_category_min_support: int,
    new_category_min_mean_conf: float,
    model_name: str,
) -> Tuple[pd.Series, pd.Series, Dict[str, object]]:
    """Refine weak labels with optional LLM relabeling for ambiguous rows.

    The LLM is called only for rows that are currently "other" or below a
    confidence threshold. New categories are proposed but promoted only when
    they pass support and confidence thresholds, to avoid taxonomy explosion.
    """
    labels = weak_labels.copy()
    conf = weak_conf.astype(float).copy()

    stats: Dict[str, object] = {
        "llm_enabled": bool(enabled),
        "model_name": model_name,
        "llm_candidates": 0,
        "llm_calls_made": 0,
        "llm_parse_failures": 0,
        "llm_nochange": 0,
        "llm_changed_existing": 0,
        "llm_new_category_proposals": 0,
        "llm_new_categories_promoted": [],
    }

    if not enabled or max_calls <= 0:
        return labels, conf, stats

    target_mask = (labels == "other") | (conf < float(low_conf_threshold))
    idxs = full_text.index[target_mask].tolist()
    stats["llm_candidates"] = int(len(idxs))
    if not idxs:
        return labels, conf, stats

    try:
        from gemini_utils import _get_client

        client = _get_client()
    except Exception as e:
        stats["llm_enabled"] = False
        stats["llm_error"] = f"LLM disabled: {e}"
        return labels, conf, stats

    valid_categories = set(BASE_PROBLEM_TYPES)
    proposed_meta: Dict[str, Dict[str, float]] = {}

    call_count = 0
    for idx in idxs:
        if call_count >= max_calls:
            break

        text = _safe_text(full_text.loc[idx])
        if not text:
            continue

        current_categories = sorted(valid_categories)
        prompt = (
            "You are classifying tech-company customer support complaints.\n"
            "Return ONLY valid JSON with keys: label, confidence, is_new_category, new_category, reason.\n"
            "Rules:\n"
            "1) Prefer one of the existing categories when possible.\n"
            "2) Use is_new_category=true only if this is a distinct recurring issue type.\n"
            "3) If ambiguous or niche, return label='other'.\n"
            f"Existing categories: {current_categories}\n"
            "Complaint text:\n"
            f"{text[:2500]}"
        )

        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            parsed = _extract_json_from_text(getattr(response, "text", ""))
        except Exception:
            parsed = None

        call_count += 1
        if not parsed:
            stats["llm_parse_failures"] = int(stats["llm_parse_failures"]) + 1
            continue

        label_raw = _normalize_category_name(str(parsed.get("label", "other")))
        conf_raw = parsed.get("confidence", 0.0)
        try:
            llm_conf = float(np.clip(float(conf_raw), 0.0, 1.0))
        except Exception:
            llm_conf = 0.0

        is_new = bool(parsed.get("is_new_category", False))
        new_cat = _normalize_category_name(str(parsed.get("new_category", ""))) if is_new else ""

        old_label = labels.loc[idx]
        old_conf = float(conf.loc[idx])

        if is_new and new_cat and len(new_cat) >= 3 and new_cat not in BASE_PROBLEM_TYPES:
            stats["llm_new_category_proposals"] = int(stats["llm_new_category_proposals"]) + 1
            rec = proposed_meta.setdefault(new_cat, {"count": 0.0, "conf_sum": 0.0})
            rec["count"] += 1.0
            rec["conf_sum"] += llm_conf
            labels.loc[idx] = new_cat
            conf.loc[idx] = max(old_conf, llm_conf)
            continue

        if label_raw in valid_categories:
            labels.loc[idx] = label_raw
            conf.loc[idx] = max(old_conf, llm_conf)
            if label_raw != old_label:
                stats["llm_changed_existing"] = int(stats["llm_changed_existing"]) + 1
            else:
                stats["llm_nochange"] = int(stats["llm_nochange"]) + 1
        else:
            stats["llm_nochange"] = int(stats["llm_nochange"]) + 1

    promoted = []
    for cat, rec in proposed_meta.items():
        mean_conf = rec["conf_sum"] / max(rec["count"], 1.0)
        if rec["count"] >= float(new_category_min_support) and mean_conf >= float(new_category_min_mean_conf):
            promoted.append(cat)

    if promoted:
        valid_categories.update(promoted)
        stats["llm_new_categories_promoted"] = sorted(promoted)

    # Any proposed category not promoted is mapped back to "other".
    non_promoted_proposed = set(proposed_meta.keys()) - set(promoted)
    if non_promoted_proposed:
        labels = labels.apply(lambda x: "other" if x in non_promoted_proposed else x)

    # Keep confidence bounded and require minimum confidence for non-other output.
    conf = conf.clip(lower=0.0, upper=1.0)
    labels = labels.where((labels == "other") | (conf >= 0.20), "other")

    stats["llm_calls_made"] = int(call_count)
    stats["final_category_set"] = sorted(valid_categories)
    return labels, conf, stats


def _method_of_moments_beta(values: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = np.clip(x, 1e-4, 1.0 - 1e-4)

    if x.size < 3:
        return 2.5, 3.0

    m = float(np.mean(x))
    v = float(np.var(x))

    max_v = m * (1.0 - m) - 1e-6
    if v <= 1e-8 or v >= max_v:
        concentration = 10.0
        return float(max(m * concentration, 1.1)), float(max((1.0 - m) * concentration, 1.1))

    common = m * (1.0 - m) / v - 1.0
    alpha = max(m * common, 1.1)
    beta = max((1.0 - m) * common, 1.1)
    return float(alpha), float(beta)


def _run_kmeans(x: np.ndarray, k: int, seed: int, iters: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    n = x.shape[0]
    if n < k:
        labels = np.arange(n) % max(1, k)
        centers = np.vstack([x[labels == i].mean(axis=0) if np.any(labels == i) else x.mean(axis=0) for i in range(k)])
        return labels, centers

    init_idx = rng.choice(n, size=k, replace=False)
    centers = x[init_idx].copy()

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        d2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d2, axis=1)

        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        for i in range(k):
            mask = labels == i
            if np.any(mask):
                centers[i] = x[mask].mean(axis=0)
            else:
                centers[i] = x[rng.randint(0, n)]

    return labels, centers


def _fit_logistic_gd(x: np.ndarray, y: np.ndarray, lr: float = 0.1, l2: float = 1e-3, epochs: int = 1200) -> np.ndarray:
    n, d = x.shape
    w = np.zeros(d, dtype=np.float64)

    for _ in range(epochs):
        z = x @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))
        grad = (x.T @ (p - y)) / n + l2 * w
        w -= lr * grad

    return w


def _build_twitter_state_table(
    threads: pd.DataFrame,
    *,
    problem_type_llm_refine: bool = False,
    problem_type_llm_low_conf_threshold: float = 0.30,
    problem_type_llm_max_calls: int = 0,
    problem_type_llm_new_category_min_support: int = 30,
    problem_type_llm_new_category_min_mean_conf: float = 0.65,
    problem_type_llm_model: str = "gemini-2.0-flash",
) -> pd.DataFrame:
    df = threads.copy()

    customer_text = df["customer_texts"].fillna("")
    brand_text = df["brand_texts"].fillna("")
    full_text = (customer_text + " " + brand_text).str.strip()

    inferred = full_text.apply(infer_problem_type_with_confidence)
    df["problem_type_weak"] = inferred.apply(lambda x: x[0])
    df["problem_type_confidence_weak"] = inferred.apply(lambda x: float(x[1]))

    # LLM refinement path is intentionally disabled for iterative regex-first calibration.
    _ = (
        problem_type_llm_refine,
        problem_type_llm_low_conf_threshold,
        problem_type_llm_max_calls,
        problem_type_llm_new_category_min_support,
        problem_type_llm_new_category_min_mean_conf,
        problem_type_llm_model,
    )

    df["problem_type"] = df["problem_type_weak"]
    df["problem_type_confidence"] = df["problem_type_confidence_weak"]
    df["problem_type_label_source"] = "weak"
    df.attrs["problem_type_llm_stats"] = {
        "llm_enabled": False,
        "status": "disabled_by_default_regex_first",
        "note": "LLM refinement is currently commented out in active flow.",
    }
    df.attrs["problem_type_other_diagnostics"] = _build_problem_type_diagnostics(
        full_text,
        df["problem_type"],
        df["problem_type_confidence"],
    )

    # Difficulty proxy combines thread length, no-reply, and escalation language.
    t_norm = np.clip(df["num_turns"].astype(float) / max(float(df["num_turns"].quantile(0.95)), 1.0), 0.0, 1.0)
    unresolved = (~df["has_brand_reply"]).astype(float)
    esc_phrase = df["has_escalation_phrase"].astype(float)
    # Low-confidence inferred problem types should carry higher uncertainty.
    low_conf = np.clip(1.0 - df["problem_type_confidence"].astype(float), 0.0, 1.0)
    df["difficulty_proxy"] = _clip01(0.40 * t_norm + 0.30 * unresolved + 0.15 * esc_phrase + 0.15 * low_conf)

    # Frustration proxy from text style + negative sentiment.
    sent = customer_text.apply(compute_sentiment).astype(float)
    exclam = customer_text.str.count("!").astype(float)
    exclam_norm = np.clip(exclam / 4.0, 0.0, 1.0)
    caps_ratio = customer_text.apply(
        lambda t: sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1)
    ).astype(float)
    caps_norm = np.clip(caps_ratio * 8.0, 0.0, 1.0)
    neg_sent = np.clip(-sent, 0.0, 1.0)
    df["frustration_proxy"] = _clip01(0.45 * neg_sent + 0.25 * exclam_norm + 0.15 * caps_norm + 0.15 * unresolved)

    # Information proxy: more turns + brand replies usually imply more information has been gathered.
    reply_ratio = df["num_brand_msgs"].astype(float) / np.maximum(df["num_customer_msgs"].astype(float), 1.0)
    reply_ratio = np.clip(reply_ratio, 0.0, 1.0)
    df["info_proxy"] = _clip01(0.6 * t_norm + 0.4 * reply_ratio)

    return df


def _estimate_personas(table: pd.DataFrame, seed: int) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    f1 = table["frustration_proxy"].to_numpy(dtype=np.float64)
    f2 = np.clip(table["difficulty_proxy"].to_numpy(dtype=np.float64), 0.0, 1.0)
    f3 = np.clip(table["num_turns"].to_numpy(dtype=np.float64) / max(float(table["num_turns"].quantile(0.95)), 1.0), 0.0, 1.0)
    f4 = (~table["has_brand_reply"]).astype(float).to_numpy()

    x = np.column_stack([f1, f2, f3, f4])
    labels, centers = _run_kmeans(x, k=4, seed=seed)

    profiles = []
    for i in range(4):
        c = centers[i]
        frustration_like = float(np.clip((c[0] + c[3]) / 2.0, 0.0, 1.0))
        persistence_like = float(np.clip((c[2] + c[1]) / 2.0, 0.0, 1.0))

        rho = float(np.clip(1.0 - 0.75 * frustration_like - 0.25 * persistence_like, 0.05, 0.95))
        sigma = float(np.clip(0.15 + 0.75 * frustration_like, 0.05, 0.95))
        tau = float(np.clip(1.0 - 0.65 * persistence_like - 0.20 * frustration_like, 0.05, 0.95))

        profiles.append((i, rho, sigma, tau))

    # Name personas by patience and tolerance pattern.
    # Highest rho => cooperative, lowest rho => impatient, lowest tau among remaining => escalation_prone.
    order_by_rho = sorted(profiles, key=lambda t: t[1])
    impatient_idx = order_by_rho[0][0]
    cooperative_idx = order_by_rho[-1][0]

    remaining = [p for p in profiles if p[0] not in (impatient_idx, cooperative_idx)]
    escalation_idx = sorted(remaining, key=lambda t: t[3])[0][0]
    silent_idx = [p[0] for p in remaining if p[0] != escalation_idx][0]

    name_by_idx = {
        cooperative_idx: "cooperative",
        impatient_idx: "impatient",
        escalation_idx: "escalation_prone",
        silent_idx: "silent_dropoff_prone",
    }

    counts = pd.Series(labels).value_counts(normalize=True)
    persona_profiles: Dict[str, Dict[str, float]] = {}
    persona_weights: Dict[str, float] = {}

    for i, rho, sigma, tau in profiles:
        name = name_by_idx[i]
        persona_profiles[name] = {
            "rho": round(rho, 4),
            "sigma": round(sigma, 4),
            "tau": round(tau, 4),
        }
        persona_weights[name] = round(float(counts.get(i, 0.0)), 4)

    # Ensure weights sum to 1 exactly in artifact.
    s = sum(persona_weights.values())
    if s > 0:
        for k in list(persona_weights.keys()):
            persona_weights[k] = round(persona_weights[k] / s, 4)

    return persona_profiles, persona_weights


def _fit_success_model_from_oasst(trees: pd.DataFrame) -> Dict[str, float]:
    df = trees.copy()
    if "avg_quality" not in df.columns:
        raise ValueError("OpenAssistant trees missing avg_quality")

    q = df["avg_quality"].astype(float)
    q_non_nan = q.dropna()
    if q_non_nan.empty:
        # CSV exports may not include quality labels; fall back to average rank.
        rank_non_nan = df.get("avg_rank", pd.Series(dtype=float)).dropna().astype(float)
        if rank_non_nan.empty:
            return {"theta0": 0.0, "theta_i": 1.0, "theta_d": 1.0, "theta_f": 1.0, "quality_threshold": 0.5}

        r = df["avg_rank"].astype(float)
        rmin = float(rank_non_nan.min())
        rmax = float(rank_non_nan.max())
        q01 = 1.0 - np.clip((r - rmin) / max(rmax - rmin, 1e-6), 0.0, 1.0)
    else:
        # Auto-detect quality scale and normalize to [0,1].
        qmax = float(q_non_nan.quantile(0.99))
        scale = 5.0 if qmax > 1.2 else 1.0
        q01 = np.clip(q / scale, 0.0, 1.0)

    quality_threshold = float(np.nanmedian(q01))
    y = (q01 >= quality_threshold).astype(float).to_numpy()

    turns = df["num_turns"].fillna(1).astype(float).to_numpy()
    t_norm = np.clip((turns - 1.0) / 10.0, 0.0, 1.0)

    # Approximate latent factors from observed tree stats.
    info_proxy = np.clip(0.55 * t_norm + 0.45 * (1.0 - np.abs(t_norm - 0.45)), 0.0, 1.0)
    difficulty_proxy = np.clip(1.0 - info_proxy + 0.20 * np.clip(t_norm - 0.6, 0.0, 1.0), 0.0, 1.0)
    frustration_proxy = np.clip(1.0 - q01.fillna(q01.mean()).to_numpy(), 0.0, 1.0)

    x = np.column_stack(
        [
            np.ones_like(info_proxy),
            info_proxy,
            difficulty_proxy,
            frustration_proxy,
        ]
    )

    w = _fit_logistic_gd(x, y)

    # Convert to canonical form p = sig(theta0 + theta_i*i - theta_d*d - theta_f*f)
    theta0 = float(w[0])
    theta_i = float(max(w[1], 0.05))
    theta_d = float(max(-w[2], 0.05))
    theta_f = float(max(-w[3], 0.05))

    return {
        "theta0": round(theta0, 6),
        "theta_i": round(theta_i, 6),
        "theta_d": round(theta_d, 6),
        "theta_f": round(theta_f, 6),
        "quality_threshold": round(quality_threshold, 6),
    }


def calibrate_task1(
    twitter_sample_frac: float,
    oasst_sample_frac: float,
    seed: int,
    twitter_max_threads: int | None,
    problem_type_llm_refine: bool = False,
    problem_type_llm_low_conf_threshold: float = 0.30,
    problem_type_llm_max_calls: int = 0,
    problem_type_llm_new_category_min_support: int = 30,
    problem_type_llm_new_category_min_mean_conf: float = 0.65,
    problem_type_llm_model: str = "gemini-2.0-flash",
) -> Dict[str, object]:
    # Twitter calibration side.
    tw_raw = load_twitter_data(sample_frac=twitter_sample_frac, random_state=seed)
    threads = reconstruct_threads(tw_raw)

    if twitter_max_threads is not None and threads.shape[0] > twitter_max_threads:
        threads = threads.sample(n=twitter_max_threads, random_state=seed).reset_index(drop=True)

    threads = add_escalation_labels(threads)
    state_table = _build_twitter_state_table(
        threads,
        problem_type_llm_refine=problem_type_llm_refine,
        problem_type_llm_low_conf_threshold=problem_type_llm_low_conf_threshold,
        problem_type_llm_max_calls=problem_type_llm_max_calls,
        problem_type_llm_new_category_min_support=problem_type_llm_new_category_min_support,
        problem_type_llm_new_category_min_mean_conf=problem_type_llm_new_category_min_mean_conf,
        problem_type_llm_model=problem_type_llm_model,
    )

    problem_prior = (
        state_table["problem_type"].value_counts(normalize=True).sort_index().to_dict()
    )
    problem_type_conf_mean = float(state_table["problem_type_confidence"].mean())
    problem_type_non_other_rate = float((state_table["problem_type"] != "other").mean())
    weak_non_other_rate = float((state_table["problem_type_weak"] != "other").mean())
    llm_stats = state_table.attrs.get("problem_type_llm_stats", {})
    other_diagnostics = state_table.attrs.get("problem_type_other_diagnostics", {})

    beta_by_problem: Dict[str, Dict[str, float]] = {}
    for ptype, group in state_table.groupby("problem_type"):
        alpha, beta = _method_of_moments_beta(group["difficulty_proxy"].to_numpy())
        beta_by_problem[ptype] = {
            "alpha": round(alpha, 6),
            "beta": round(beta, 6),
            "n": int(group.shape[0]),
        }

    persona_profiles, persona_weights = _estimate_personas(state_table, seed=seed)

    # OpenAssistant calibration side.
    # Use CSV fallback to avoid requiring fastparquet in local setups.
    oa_raw = load_openassistant_data(use_parquet=False, sample_frac=oasst_sample_frac, random_state=seed)
    trees = build_conversation_trees(oa_raw)
    success_model = _fit_success_model_from_oasst(trees)

    # Tier is not explicit in public datasets; keep explicit policy prior for now.
    tier_prior = {
        "Free": 0.60,
        "Pro": 0.25,
        "Business+": 0.12,
        "Enterprise": 0.03,
    }

    tier_value_weight = {
        "Free": 0.00,
        "Pro": 0.35,
        "Business+": 0.65,
        "Enterprise": 1.00,
    }

    artifact = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "data_sources": {
            "twitter_path": DATA_PATHS["twitter"],
            "openassistant_train_path": DATA_PATHS["openassistant_train"],
            "openassistant_val_path": DATA_PATHS["openassistant_val"],
            "twitter_rows_loaded": int(tw_raw.shape[0]),
            "twitter_threads_built": int(threads.shape[0]),
            "openassistant_rows_loaded": int(oa_raw.shape[0]),
            "openassistant_trees_built": int(trees.shape[0]),
        },
        "task1_parameters": {
            "tier_prior": tier_prior,
            "tier_value_weight": tier_value_weight,
            "problem_type_prior": {k: round(float(v), 6) for k, v in problem_prior.items()},
            "problem_type_inference_stats": {
                "mean_confidence": round(problem_type_conf_mean, 6),
                "non_other_rate": round(problem_type_non_other_rate, 6),
                "weak_non_other_rate": round(weak_non_other_rate, 6),
                "llm_refinement": llm_stats,
                "other_diagnostics": other_diagnostics,
            },
            "difficulty_beta_by_problem_type": beta_by_problem,
            "persona_profiles": persona_profiles,
            "persona_weights": persona_weights,
            "success_model": success_model,
            "reset_dynamic_defaults": {
                "frustration_beta": {"alpha": 1.5, "beta": 8.0},
                "information_beta": {"alpha": 1.2, "beta": 6.0},
                "turn_count": 0,
                "resolved": 0,
                "failed_streak": 0,
            },
        },
        "notes": [
            "problem_type inferred using regex weak supervision in this run",
            "llm refinement path is intentionally disabled; use other_diagnostics output for iterative regex updates",
            "tier and customer value are externally assumed business priors",
            "success model fitted from OASST quality labels and proxy latent features",
            "all fitted values are intended as initialization and should be re-estimated in later calibration passes",
        ],
    }

    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Task 1 hidden-state parameters from real datasets.")
    parser.add_argument(
        "--twitter-sample-frac",
        type=float,
        default=1.0,
        help="Fraction of raw Twitter rows to load; use 1.0 to preserve thread structure",
    )
    parser.add_argument(
        "--twitter-max-threads",
        type=int,
        default=50000,
        help="Optional max threads after reconstruction for calibration speed",
    )
    parser.add_argument("--oasst-sample-frac", type=float, default=0.30, help="Fraction of OASST rows to load")
    parser.add_argument(
        "--problem-type-llm-refine",
        action="store_true",
        help="Enable LLM refinement for problem_type on 'other'/low-confidence rows.",
    )
    parser.add_argument(
        "--problem-type-llm-low-conf-threshold",
        type=float,
        default=0.30,
        help="Rows with confidence below this are candidates for LLM refinement.",
    )
    parser.add_argument(
        "--problem-type-llm-max-calls",
        type=int,
        default=0,
        help="Maximum LLM calls for problem_type refinement (0 disables calls).",
    )
    parser.add_argument(
        "--problem-type-llm-new-category-min-support",
        type=int,
        default=30,
        help="Minimum LLM proposals required to promote a new category.",
    )
    parser.add_argument(
        "--problem-type-llm-new-category-min-mean-conf",
        type=float,
        default=0.65,
        help="Minimum average LLM confidence required to promote a new category.",
    )
    parser.add_argument(
        "--problem-type-llm-model",
        type=str,
        default="gemini-2.0-flash",
        help="LLM model name used for problem_type refinement.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(OUTPUT_DIR) / "task1_calibration.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    artifact = calibrate_task1(
        twitter_sample_frac=args.twitter_sample_frac,
        oasst_sample_frac=args.oasst_sample_frac,
        seed=args.seed,
        twitter_max_threads=args.twitter_max_threads,
        problem_type_llm_refine=args.problem_type_llm_refine,
        problem_type_llm_low_conf_threshold=args.problem_type_llm_low_conf_threshold,
        problem_type_llm_max_calls=args.problem_type_llm_max_calls,
        problem_type_llm_new_category_min_support=args.problem_type_llm_new_category_min_support,
        problem_type_llm_new_category_min_mean_conf=args.problem_type_llm_new_category_min_mean_conf,
        problem_type_llm_model=args.problem_type_llm_model,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print(f"Saved Task 1 calibration artifact to: {out_path}")
    print(json.dumps(artifact["task1_parameters"]["success_model"], indent=2))


if __name__ == "__main__":
    main()
