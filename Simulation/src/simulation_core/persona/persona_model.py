from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from simulation_core.config import LOW_CONFIDENCE_THRESHOLD

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # pragma: no cover - runtime optional
    torch = None
    nn = None
    optim = None


ACTION_SPACE_7 = [
    "Ask_for_Information",
    "Provide_Solution",
    "Affective_Repair",
    "Escalate_to_Human",
    "Close_with_Feedback",
    "Proactive_Update",
    "Set_Expectation",
]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTION_SPACE_7)}


class CVAE(nn.Module):
    def __init__(self, input_dim: int, condition_dim: int, z_dim: int = 8, output_dim: int = 2):
        super().__init__()
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(32, z_dim)
        self.log_var_layer = nn.Linear(32, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim + condition_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

        self.escalation_head = nn.Linear(z_dim, 1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu_layer(h), self.log_var_layer(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, cond], dim=-1))

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z, cond)
        esc = torch.sigmoid(self.escalation_head(z))
        return recon, mu, log_var, z, esc


@dataclass
class PersonaState:
    model: CVAE
    z: torch.Tensor

    def update(self, action_taken: str, response_sentiment: float) -> None:
        if torch is None:
            return
        action_vec = torch.zeros((1, len(ACTION_SPACE_7)), dtype=torch.float32)
        if action_taken in ACTION_TO_ID:
            action_vec[0, ACTION_TO_ID[action_taken]] = 1.0
        cond = torch.cat(
            [
                torch.tensor([[0.5, response_sentiment, 0.5]], dtype=torch.float32),
                action_vec,
            ],
            dim=1,
        )
        with torch.no_grad():
            delta = self.model.decode(self.z, cond)
            noise = 0.05 * torch.ones_like(self.z) * torch.randn_like(self.z)
            self.z = self.z + delta.mean() * 0.01 + noise

    def frustration_estimate(self) -> float:
        if torch is None:
            return 0.5
        with torch.no_grad():
            return float(torch.sigmoid(self.z.mean()).item())

    def escalation_risk(self) -> float:
        if torch is None:
            return 0.1
        with torch.no_grad():
            return float(torch.sigmoid(self.model.escalation_head(self.z)).mean().item())


def _conv_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for conv_id, g in df.groupby("conv_id"):
        customer = g[g["speaker_role"] == "customer"]
        agent = g[g["speaker_role"] == "agent"]

        fr = customer["frustration_score"].dropna().astype(float)
        se = customer["sentiment_score"].dropna().astype(float)

        has_escalation = int((agent["action_label"] == "Escalate_to_Human").any())
        domain = g["domain_source"].iloc[0]
        domain_twitter = 1.0 if domain == "twitter_cs" else 0.0
        domain_open = 1.0 - domain_twitter

        rows.append(
            {
                "conv_id": conv_id,
                "mean_frustration": float(fr.mean()) if not fr.empty else 0.5,
                "std_frustration": float(fr.std()) if len(fr) > 1 else 0.0,
                "mean_sentiment": float(se.mean()) if not se.empty else 0.0,
                "std_sentiment": float(se.std()) if len(se) > 1 else 0.0,
                "conv_length_norm": float(g["conv_length"].iloc[0]) / 20.0,
                "escalated": has_escalation,
                "domain_twitter": domain_twitter,
                "domain_openassistant": domain_open,
            }
        )
    return pd.DataFrame(rows)


def _turn_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in df.groupby("conv_id"):
        g = g.sort_values("turn_index")
        for i in range(len(g) - 1):
            cur = g.iloc[i]
            nxt = g.iloc[i + 1]
            if pd.isna(cur.get("frustration_score")) or pd.isna(cur.get("sentiment_score")):
                continue
            if pd.isna(nxt.get("frustration_score")) or pd.isna(nxt.get("sentiment_score")):
                continue

            action = cur.get("action_label") if cur.get("speaker_role") == "agent" else "Provide_Solution"
            if action not in ACTION_TO_ID:
                action = "Provide_Solution"

            action_one_hot = np.zeros(len(ACTION_SPACE_7), dtype=float)
            action_one_hot[ACTION_TO_ID[action]] = 1.0

            rows.append(
                {
                    "conv_id": cur["conv_id"],
                    "frustration": float(cur["frustration_score"]),
                    "sentiment": float(cur["sentiment_score"]),
                    "turn_norm": float(cur["turn_index"]) / max(float(cur["conv_length"]), 1.0),
                    "action_one_hot": action_one_hot,
                    "delta_frustration": float(nxt["frustration_score"] - cur["frustration_score"]),
                    "delta_sentiment": float(nxt["sentiment_score"] - cur["sentiment_score"]),
                }
            )
    return pd.DataFrame(rows)


def _elbo_loss(recon, target, mu, log_var, beta: float = 0.5):
    recon_loss = ((recon - target) ** 2).mean()
    kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl, recon_loss, kl


def train_persona_cvae(labeled_df: pd.DataFrame, out_dir: Path) -> Dict[str, object]:
    if torch is None:
        raise RuntimeError("PyTorch is required for persona model training.")

    out_dir.mkdir(parents=True, exist_ok=True)

    filtered = labeled_df[labeled_df["annotator_confidence"].fillna(0.0) >= LOW_CONFIDENCE_THRESHOLD].copy()
    conv = _conv_features(filtered)
    turns = _turn_training_rows(filtered)

    if len(conv) < 20 or len(turns) < 50:
        raise ValueError("Not enough high-confidence data to train CVAE persona model.")

    conv_train, conv_tmp = train_test_split(
        conv,
        test_size=0.2,
        random_state=42,
        stratify=conv[["domain_twitter", "domain_openassistant"]],
    )
    conv_val, conv_test = train_test_split(conv_tmp, test_size=0.5, random_state=42)

    train_conv_ids = set(conv_train["conv_id"])
    val_conv_ids = set(conv_val["conv_id"])

    train_turns = turns[turns["conv_id"].isin(train_conv_ids)]
    val_turns = turns[turns["conv_id"].isin(val_conv_ids)]

    input_cols = [
        "mean_frustration",
        "std_frustration",
        "mean_sentiment",
        "std_sentiment",
        "conv_length_norm",
        "escalated",
        "domain_twitter",
        "domain_openassistant",
    ]
    model = CVAE(input_dim=len(input_cols), condition_dim=3 + len(ACTION_SPACE_7), z_dim=8, output_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    conv_lookup = conv.set_index("conv_id")

    def _to_batch(df: pd.DataFrame):
        x_rows = []
        c_rows = []
        y_rows = []
        for _, r in df.iterrows():
            conv_id = r["conv_id"]
            x_rows.append(conv_lookup.loc[conv_id, input_cols].values.astype(np.float32))
            cond = np.concatenate(
                [
                    np.array([r["frustration"], r["sentiment"], r["turn_norm"]], dtype=np.float32),
                    r["action_one_hot"].astype(np.float32),
                ]
            )
            c_rows.append(cond)
            y_rows.append(np.array([r["delta_frustration"], r["delta_sentiment"]], dtype=np.float32))
        return (
            torch.tensor(np.stack(x_rows), dtype=torch.float32),
            torch.tensor(np.stack(c_rows), dtype=torch.float32),
            torch.tensor(np.stack(y_rows), dtype=torch.float32),
        )

    x_train, c_train, y_train = _to_batch(train_turns)
    x_val, c_val, y_val = _to_batch(val_turns if not val_turns.empty else train_turns.head(8))

    best_val = float("inf")
    best_path = out_dir / "persona_cvae.pt"

    for _ in range(50):
        model.train()
        optimizer.zero_grad()
        recon, mu, log_var, _, _ = model(x_train, c_train)
        loss, _, _ = _elbo_loss(recon, y_train, mu, log_var, beta=0.5)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_recon, val_mu, val_log_var, _, _ = model(x_val, c_val)
            val_loss, _, _ = _elbo_loss(val_recon, y_val, val_mu, val_log_var, beta=0.5)
            score = float(val_loss.item())
            if score < best_val:
                best_val = score
                torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location="cpu"))
    model.eval()

    # Validation task: escalation prediction from first 3 turns.
    conv_with_score = []
    for conv_id, g in filtered.groupby("conv_id"):
        g = g.sort_values("turn_index").head(3)
        if g.empty:
            continue
        fr = g["frustration_score"].dropna().astype(float)
        se = g["sentiment_score"].dropna().astype(float)
        if fr.empty or se.empty:
            continue

        base = conv_lookup.loc[conv_id, input_cols].values.astype(np.float32)
        with torch.no_grad():
            mu, _ = model.encode(torch.tensor(base).unsqueeze(0))
            z_score = float(torch.sigmoid(model.escalation_head(mu)).mean().item())

        conv_all = filtered[filtered["conv_id"] == conv_id]
        y = int((conv_all["action_label"] == "Escalate_to_Human").any())

        conv_with_score.append(
            {
                "conv_id": conv_id,
                "y": y,
                "cvae_score": z_score,
                "mean_frustration_baseline": float(fr.mean()),
                "gmm_feature": float(fr.mean() - se.mean()),
            }
        )

    eval_df = pd.DataFrame(conv_with_score)
    if eval_df["y"].nunique() < 2:
        auc_cvae = 0.5
        auc_gmm = 0.5
        auc_mean_fr = 0.5
    else:
        auc_cvae = float(roc_auc_score(eval_df["y"], eval_df["cvae_score"]))

        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm_pred = gmm.fit_predict(eval_df[["gmm_feature"]])
        auc_gmm = float(roc_auc_score(eval_df["y"], gmm_pred))

        auc_mean_fr = float(roc_auc_score(eval_df["y"], eval_df["mean_frustration_baseline"]))

    result = {
        "auc_cvae": auc_cvae,
        "auc_gmm": auc_gmm,
        "auc_mean_frustration": auc_mean_fr,
        "best_val_loss": best_val,
        "n_conversations_eval": int(len(eval_df)),
    }

    out_json = out_dir / "persona_validation.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
