from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler


FINAL_FEATURES = [
    "relative_turn_count",
    "total_values_provided",
    "avg_values_per_action_turn",
    "forward_progress_rate",
    "resolution_flag",
    "escalation_flag",
]


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def _compute_stability(x_scaled: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in [3, 4, 5, 6]:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(x_scaled)
        sil = float(silhouette_score(x_scaled, labels))
        sil_by_cluster = (
            pd.DataFrame({"cluster_id": labels, "sil": silhouette_samples(x_scaled, labels)})
            .groupby("cluster_id", as_index=False)["sil"].mean()
        )
        rows.append(
            {
                "k": k,
                "silhouette_overall": sil,
                "inertia": float(model.inertia_),
                "min_cluster_silhouette": float(sil_by_cluster["sil"].min()),
                "max_cluster_silhouette": float(sil_by_cluster["sil"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def _assign_personas(profile: pd.DataFrame) -> dict[int, str]:
    p = profile.copy().set_index("cluster_id")
    ids = sorted(p.index.tolist())
    label_map: dict[int, str] = {}

    esc_candidates = p[p["escalation_flag"] >= 0.999].index.tolist()
    esc_id = int(esc_candidates[0]) if esc_candidates else int(p["escalation_flag"].idxmax())
    label_map[esc_id] = "escalation_prone"

    remaining = [cid for cid in ids if cid != esc_id]
    p_rem = p.loc[remaining]

    silent_candidates = p_rem[(p_rem["resolution_flag"] <= 0.001) & (p_rem["escalation_flag"] <= 0.001)].index.tolist()
    if silent_candidates:
        silent_id = int(p.loc[silent_candidates, "relative_turn_count"].idxmin())
    else:
        silent_id = int((p_rem["resolution_flag"] + p_rem["escalation_flag"]).idxmin())
    label_map[silent_id] = "silent_dropout"

    remaining = [cid for cid in ids if cid not in label_map]
    p_rem = p.loc[remaining]
    resolver_candidates = p_rem[(p_rem["resolution_flag"] >= 0.999) & (p_rem["escalation_flag"] <= 0.001)].index.tolist()
    if len(resolver_candidates) < 2:
        resolver_candidates = p_rem.sort_values(["resolution_flag", "escalation_flag"], ascending=[False, True]).index.tolist()[:2]

    high_id = int(p.loc[resolver_candidates, "total_values_provided"].idxmax())
    low_id = int(p.loc[resolver_candidates, "total_values_provided"].idxmin())

    label_map[high_id] = "high_engagement_resolver"
    label_map[low_id] = "low_engagement_resolver"

    leftovers = [cid for cid in ids if cid not in label_map]
    for cid in leftovers:
        # Defensive fallback; this should not happen for k=4.
        label_map[cid] = "low_engagement_resolver"

    return label_map


def _compute_persona_params(clustered: pd.DataFrame, persona_by_cluster: dict[int, str]) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    warnings: list[str] = []

    ranges = {
        "high_engagement_resolver": {"rho": (0.65, 0.80), "sigma": (0.20, 0.35), "tau": (0.65, 0.80)},
        "low_engagement_resolver": {"rho": (0.45, 0.65), "sigma": (0.25, 0.40), "tau": (0.40, 0.60)},
        "silent_dropout": {"rho": (0.15, 0.35), "sigma": (0.40, 0.60), "tau": (0.15, 0.35)},
        "escalation_prone": {"rho": (0.30, 0.55), "sigma": (0.65, 0.85), "tau": (0.15, 0.30)},
    }

    for cid, g in clustered.groupby("cluster_id"):
        means = g[FINAL_FEATURES].mean()
        stds = g[FINAL_FEATURES].std(ddof=0)

        res = float(means["resolution_flag"])
        esc = float(means["escalation_flag"])
        rtc_norm = _clip(float(means["relative_turn_count"]) / 2.0, 0.0, 1.0)
        fpr_norm = _clip(float(means["forward_progress_rate"]) / 0.5, 0.0, 1.0)
        tval_norm = _clip(float(means["total_values_provided"]) / 10.0, 0.0, 1.0)
        avpt_norm = _clip(float(means["avg_values_per_action_turn"]) / 2.0, 0.0, 1.0)

        rho_mean = _clip(0.40 * tval_norm + 0.30 * res + 0.20 * rtc_norm + 0.10 * (1.0 - esc), 0.15, 0.90)
        sigma_mean = _clip(0.15 + 0.45 * esc + 0.25 * (1.0 - avpt_norm) + 0.15 * (1.0 - res), 0.15, 0.90)
        tau_mean = _clip(0.5 * res + 0.3 * tval_norm + 0.2 * fpr_norm, 0.10, 0.90)

        rho_std = _clip(
            0.4 * float(stds["resolution_flag"]) + 0.3 * float(stds["relative_turn_count"]) / 2.0 + 0.2 * float(stds["forward_progress_rate"]) / 0.5,
            0.05,
            0.15,
        )
        sigma_std = _clip(
            0.5 * float(stds["escalation_flag"]) + 0.3 * float(stds["avg_values_per_action_turn"]) / 2.0 + 0.2 * float(stds["resolution_flag"]),
            0.05,
            0.15,
        )
        tau_std = _clip(
            0.5 * float(stds["resolution_flag"]) + 0.3 * float(stds["total_values_provided"]) / 10.0 + 0.2 * float(stds["forward_progress_rate"]) / 0.5,
            0.05,
            0.15,
        )

        label = persona_by_cluster[int(cid)]
        expected = ranges.get(label)
        if expected is not None:
            for key, val in [("rho", rho_mean), ("sigma", sigma_mean), ("tau", tau_mean)]:
                lo, hi = expected[key]
                if val < lo or val > hi:
                    warnings.append(
                        f"WARNING: {label} {key}={val:.3f} outside expected [{lo:.2f}, {hi:.2f}] (inputs: res={res:.3f}, esc={esc:.3f}, rtc_norm={rtc_norm:.3f}, fpr_norm={fpr_norm:.3f}, tval_norm={tval_norm:.3f}, avpt_norm={avpt_norm:.3f})"
                    )

        rows.append(
            {
                "cluster_id": int(cid),
                "label": label,
                "rho_mean": rho_mean,
                "rho_std": rho_std,
                "sigma_mean": sigma_mean,
                "sigma_std": sigma_std,
                "tau_mean": tau_mean,
                "tau_std": tau_std,
            }
        )

    return pd.DataFrame(rows), warnings


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    phase1_dir = root / "simulation" / "artifacts" / "phase 1"
    out_dir = root / "simulation" / "artifacts" / "phase 5"
    trial_dir = root / "simulation" / "artifacts" / "phase5_trial_1"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_path = phase1_dir / "extract5_behavioral_feature_table.csv"
    base = pd.read_csv(base_path)

    # Step 1: canonical final Extract 5
    drop_cols = [c for c in ["turns_before_first_value", "member_level_encoded", "clarification_turns"] if c in base.columns]
    if drop_cols:
        base = base.drop(columns=drop_cols)

    missing = [c for c in FINAL_FEATURES if c not in base.columns]
    if missing:
        raise RuntimeError(f"Missing required final features in base extract5: {missing}")

    final_extract5 = base[["convo_id", "subflow", *FINAL_FEATURES]].copy()
    for c in FINAL_FEATURES:
        final_extract5[c] = pd.to_numeric(final_extract5[c], errors="coerce")

    null_total = int(final_extract5[FINAL_FEATURES].isna().sum().sum())
    if null_total != 0:
        raise RuntimeError(f"Null values found in final feature table: {null_total}")

    final_extract5_path = out_dir / "extract5_behavioral_feature_table_final.csv"
    final_extract5.to_csv(final_extract5_path, index=False)

    print("Final Extract 5 summary stats (mean/std/min/max):")
    print(final_extract5[FINAL_FEATURES].agg(["mean", "std", "min", "max"]).to_string())
    print(f"Zero-null check: {null_total == 0}")
    print(f"Saved: {final_extract5_path}")

    # Step 2: fit clustering
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(final_extract5[FINAL_FEATURES].to_numpy())

    stability = _compute_stability(x_scaled)
    stability_path = out_dir / "phase5_k_stability_table_final.csv"
    stability.to_csv(stability_path, index=False)

    k_final = 4
    model = KMeans(n_clusters=k_final, random_state=42, n_init=20)
    labels = model.fit_predict(x_scaled)

    labeled = final_extract5.copy()
    labeled["cluster_id"] = labels
    labeled_path = out_dir / "extract5_with_clusters_final.csv"
    labeled.to_csv(labeled_path, index=False)
    print(f"Saved: {labeled_path}")

    # Step 3: dynamic persona mapping
    profile = (
        labeled.groupby("cluster_id", as_index=False)
        .agg(
            size=("convo_id", "count"),
            relative_turn_count=("relative_turn_count", "mean"),
            total_values_provided=("total_values_provided", "mean"),
            avg_values_per_action_turn=("avg_values_per_action_turn", "mean"),
            forward_progress_rate=("forward_progress_rate", "mean"),
            resolution_flag=("resolution_flag", "mean"),
            escalation_flag=("escalation_flag", "mean"),
        )
        .sort_values("cluster_id")
        .reset_index(drop=True)
    )
    profile["frequency"] = profile["size"] / len(labeled)

    persona_by_cluster = _assign_personas(profile)
    labeled["persona_label"] = labeled["cluster_id"].map(persona_by_cluster)

    # Step 4: compute (rho, sigma, tau) from formulas
    param_df, warnings = _compute_persona_params(labeled, persona_by_cluster)

    profile = profile.merge(param_df, on="cluster_id", how="left")
    profile["label"] = profile["cluster_id"].map(persona_by_cluster)

    order = ["high_engagement_resolver", "low_engagement_resolver", "silent_dropout", "escalation_prone"]
    profile["label_order"] = profile["label"].map({k: i for i, k in enumerate(order)})
    profile = profile.sort_values("label_order").drop(columns=["label_order"]).reset_index(drop=True)

    print("\nComputed persona parameters (from formulas):")
    print(
        profile[
            [
                "label",
                "cluster_id",
                "rho_mean",
                "rho_std",
                "sigma_mean",
                "sigma_std",
                "tau_mean",
                "tau_std",
            ]
        ].to_string(index=False)
    )
    for w in warnings:
        print(w)

    # Step 5: persona calibration artifact
    sil_k4 = float(stability.loc[stability["k"] == 4, "silhouette_overall"].iloc[0])
    sil_k3 = float(stability.loc[stability["k"] == 3, "silhouette_overall"].iloc[0])
    sil_k5 = float(stability.loc[stability["k"] == 5, "silhouette_overall"].iloc[0])
    min_sil_k4 = float(stability.loc[stability["k"] == 4, "min_cluster_silhouette"].iloc[0])

    personas = []
    for _, r in profile.iterrows():
        label = str(r["label"])
        cid = int(r["cluster_id"])
        personas.append(
            {
                "label": label,
                "cluster_id": cid,
                "size": int(r["size"]),
                "frequency": float(r["frequency"]),
                "rho_mean": float(r["rho_mean"]),
                "rho_std": float(r["rho_std"]),
                "sigma_mean": float(r["sigma_mean"]),
                "sigma_std": float(r["sigma_std"]),
                "tau_mean": float(r["tau_mean"]),
                "tau_std": float(r["tau_std"]),
                "feature_means": {
                    "relative_turn_count": float(r["relative_turn_count"]),
                    "total_values_provided": float(r["total_values_provided"]),
                    "avg_values_per_action_turn": float(r["avg_values_per_action_turn"]),
                    "forward_progress_rate": float(r["forward_progress_rate"]),
                    "resolution_flag": float(r["resolution_flag"]),
                    "escalation_flag": float(r["escalation_flag"]),
                },
            }
        )

    pi_persona = {p["label"]: p["frequency"] for p in personas}

    payload = {
        "clustering_notes": {
            "features_used": FINAL_FEATURES,
            "features_excluded": {
                "member_level_encoded": "business variable, belongs in reward function only",
                "turns_before_first_value": "mean difference of 0.24 across resolver clusters, not behaviorally meaningful",
                "clarification_turns": "short text length proxy too noisy, captures acknowledgements not confusion",
            },
            "k": 4,
            "silhouette_score": float(sil_k4),
            "rationale": f"k=4 chosen over k=3 (sil={sil_k3:.3f}) due to clear elbow at k=5 (sil={sil_k5:.3f}) and superior min-cluster silhouette ({min_sil_k4:.3f} vs {float(stability.loc[stability['k']==3, 'min_cluster_silhouette'].iloc[0]):.3f}). Resolver split reflects task complexity (high vs low information exchange) rather than pure patience/cooperation - relabeled accordingly.",
        },
        "personas": personas,
        "pi_persona": pi_persona,
    }

    persona_path = out_dir / "persona_profiles.json"
    persona_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    profile.to_csv(out_dir / "phase5_cluster_profiles.csv", index=False)

    # Step 6: validation checks
    by_label = {r["label"]: r for _, r in profile.iterrows()}
    checks = []

    def add_check(name: str, holds: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if holds else "FAIL", "detail": detail})

    max_total_values_label = str(profile.sort_values("total_values_provided", ascending=False).iloc[0]["label"])
    add_check(
        "1) high_engagement_resolver has highest total_values_provided",
        max_total_values_label == "high_engagement_resolver",
        f"observed_highest={max_total_values_label}",
    )

    resolver_vals = {
        "high_engagement_resolver": float(by_label["high_engagement_resolver"]["total_values_provided"]),
        "low_engagement_resolver": float(by_label["low_engagement_resolver"]["total_values_provided"]),
    }
    add_check(
        "2) low_engagement_resolver has lower total_values among resolver clusters",
        resolver_vals["low_engagement_resolver"] < resolver_vals["high_engagement_resolver"],
        f"low={resolver_vals['low_engagement_resolver']:.4f}, high={resolver_vals['high_engagement_resolver']:.4f}",
    )

    add_check(
        "3) escalation_prone escalation_flag mean == 1.0",
        abs(float(by_label["escalation_prone"]["escalation_flag"]) - 1.0) < 1e-9,
        f"escalation_mean={float(by_label['escalation_prone']['escalation_flag']):.4f}",
    )

    add_check(
        "4) silent_dropout has resolution=0 and escalation=0",
        abs(float(by_label["silent_dropout"]["resolution_flag"])) < 1e-9 and abs(float(by_label["silent_dropout"]["escalation_flag"])) < 1e-9,
        f"resolution_mean={float(by_label['silent_dropout']['resolution_flag']):.4f}, escalation_mean={float(by_label['silent_dropout']['escalation_flag']):.4f}",
    )

    add_check(
        "5) both resolver clusters have resolution=1 and escalation=0",
        abs(float(by_label["high_engagement_resolver"]["resolution_flag"]) - 1.0) < 1e-9
        and abs(float(by_label["high_engagement_resolver"]["escalation_flag"])) < 1e-9
        and abs(float(by_label["low_engagement_resolver"]["resolution_flag"]) - 1.0) < 1e-9
        and abs(float(by_label["low_engagement_resolver"]["escalation_flag"])) < 1e-9,
        (
            f"high(res={float(by_label['high_engagement_resolver']['resolution_flag']):.4f},esc={float(by_label['high_engagement_resolver']['escalation_flag']):.4f}), "
            f"low(res={float(by_label['low_engagement_resolver']['resolution_flag']):.4f},esc={float(by_label['low_engagement_resolver']['escalation_flag']):.4f})"
        ),
    )

    add_check(
        "6) rho(high_engagement_resolver) > rho(low_engagement_resolver)",
        float(by_label["high_engagement_resolver"]["rho_mean"]) > float(by_label["low_engagement_resolver"]["rho_mean"]),
        f"rho_high={float(by_label['high_engagement_resolver']['rho_mean']):.4f}, rho_low={float(by_label['low_engagement_resolver']['rho_mean']):.4f}",
    )

    add_check(
        "7) sigma(escalation_prone) > sigma(silent_dropout)",
        float(by_label["escalation_prone"]["sigma_mean"]) > float(by_label["silent_dropout"]["sigma_mean"]),
        f"sigma_esc={float(by_label['escalation_prone']['sigma_mean']):.4f}, sigma_silent={float(by_label['silent_dropout']['sigma_mean']):.4f}",
    )

    add_check(
        "8) tau(high_engagement_resolver) > tau(silent_dropout)",
        float(by_label["high_engagement_resolver"]["tau_mean"]) > float(by_label["silent_dropout"]["tau_mean"]),
        f"tau_high={float(by_label['high_engagement_resolver']['tau_mean']):.4f}, tau_silent={float(by_label['silent_dropout']['tau_mean']):.4f}",
    )

    checks_df = pd.DataFrame(checks)
    checks_path = out_dir / "phase5_validation_checks_final.csv"
    checks_df.to_csv(checks_path, index=False)

    print("\nValidation checks:")
    print(checks_df.to_string(index=False))

    all_pass = bool((checks_df["status"] == "PASS").all())
    if not all_pass:
        failed = checks_df[checks_df["status"] == "FAIL"]
        raise RuntimeError("One or more Phase 5 validation checks failed:\n" + failed.to_string(index=False))

    # Step 7: clean up trial directory after successful validation.
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
        print(f"Deleted trial directory: {trial_dir}")
    else:
        print("Trial directory already absent; nothing to delete.")

    print("\nSaved files:")
    print(final_extract5_path)
    print(labeled_path)
    print(stability_path)
    print(persona_path)
    print(checks_path)
    print("\npersona_profiles.json contents:")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
