from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from simulation_core.config import ACTION_SPACE_7, LOW_CONFIDENCE_THRESHOLD, TIER_VALUE_WEIGHT
from simulation_core.env.pomdp_environment import CustomerSupportPOMDP
from simulation_core.rewards.reward_model import load_reward_model

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # pragma: no cover
    torch = None
    nn = None
    optim = None


ACTION_TO_ID = {a: i for i, a in enumerate(ACTION_SPACE_7)}


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def _row_to_obs(r: pd.Series) -> np.ndarray:
    fr = float(r.get("frustration_score", 0.5) if pd.notna(r.get("frustration_score", np.nan)) else 0.5)
    se = float(r.get("sentiment_score", 0.0) if pd.notna(r.get("sentiment_score", np.nan)) else 0.0)
    ti = float(r.get("turn_index", 1)) / max(float(r.get("conv_length", 10)), 1.0)
    conf = float(r.get("annotator_confidence", 0.0) if pd.notna(r.get("annotator_confidence", np.nan)) else 0.0)
    tier = str(r.get("tier", "Pro"))
    tier_weight = float(TIER_VALUE_WEIGHT.get(tier, 0.4))
    return np.array([fr, se, ti, conf, tier_weight], dtype=np.float32)


def behavior_cloning_train(labeled_df: pd.DataFrame, out_dir: Path, epochs: int = 40) -> Dict[str, float]:
    if torch is None:
        raise RuntimeError("PyTorch is required for BC/CQL/PPO pipeline.")

    out_dir.mkdir(parents=True, exist_ok=True)
    data = labeled_df[
        (labeled_df["speaker_role"] == "agent")
        & (labeled_df["annotator_confidence"].fillna(0.0) >= LOW_CONFIDENCE_THRESHOLD)
        & (labeled_df["action_label"].isin(ACTION_SPACE_7))
    ].copy()

    if data.empty:
        raise ValueError("No high-confidence agent turns found for behavior cloning.")

    X = np.stack([_row_to_obs(r) for _, r in data.iterrows()])
    y = np.array([ACTION_TO_ID[a] for a in data["action_label"]], dtype=np.int64)

    idx = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(idx)
    cut = int(len(X) * 0.8)
    train_idx, val_idx = idx[:cut], idx[cut:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.long)

    model = PolicyNet(obs_dim=X.shape[1], n_actions=len(ACTION_SPACE_7))
    opt = optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_path = out_dir / "bc_policy.pt"

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_train)
        loss = ce(logits, y_train)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(X_val).argmax(dim=1)
            acc = float((pred == y_val).float().mean().item()) if len(y_val) else 0.0
            if acc >= best_acc:
                best_acc = acc
                torch.save(model.state_dict(), best_path)

    return {"bc_val_accuracy": best_acc, "bc_checkpoint": str(best_path)}


def cql_lite_train(
    labeled_df: pd.DataFrame,
    reward_weights_path: Path,
    bc_checkpoint: Path,
    out_dir: Path,
    gradient_steps: int = 200,
) -> Dict[str, float]:
    if torch is None:
        raise RuntimeError("PyTorch is required for CQL training.")

    out_dir.mkdir(parents=True, exist_ok=True)
    model = PolicyNet(obs_dim=5, n_actions=len(ACTION_SPACE_7))
    if bc_checkpoint.exists():
        model.load_state_dict(torch.load(bc_checkpoint, map_location="cpu"), strict=False)

    target = PolicyNet(obs_dim=5, n_actions=len(ACTION_SPACE_7))
    target.load_state_dict(model.state_dict())

    opt = optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    reward_model = load_reward_model(reward_weights_path)

    data = labeled_df[(labeled_df["speaker_role"] == "agent") & (labeled_df["action_label"].isin(ACTION_SPACE_7))].copy()
    X = np.stack([_row_to_obs(r) for _, r in data.iterrows()])
    A = np.array([ACTION_TO_ID[a] for a in data["action_label"]], dtype=np.int64)

    rewards = []
    for _, r in data.iterrows():
        state = {
            "frustration": float(r.get("frustration_score", 0.5) if pd.notna(r.get("frustration_score", np.nan)) else 0.5),
            "sentiment": float(r.get("sentiment_score", 0.0) if pd.notna(r.get("sentiment_score", np.nan)) else 0.0),
            "turn_index_norm": float(r.get("turn_index", 1)) / max(float(r.get("conv_length", 10)), 1.0),
            "tier": str(r.get("tier", "Pro")),
            "escalation_risk": float(r.get("frustration_score", 0.5) if pd.notna(r.get("frustration_score", np.nan)) else 0.5),
        }
        rewards.append(reward_model.score(state, str(r["action_label"])))

    R = np.array(rewards, dtype=np.float32)

    X_t = torch.tensor(X, dtype=torch.float32)
    A_t = torch.tensor(A, dtype=torch.long)
    R_t = torch.tensor(R, dtype=torch.float32)

    gamma = 0.99
    alpha = 1.0

    for _ in range(gradient_steps):
        q = model(X_t)
        q_sa = q.gather(1, A_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = target(X_t).max(dim=1).values
            y = R_t + gamma * q_next

        bellman = mse(q_sa, y)
        conservative = (torch.logsumexp(q, dim=1) - q_sa).mean()
        loss = bellman + alpha * conservative

        opt.zero_grad()
        loss.backward()
        opt.step()

    cql_path = out_dir / "cql_qnet.pt"
    torch.save(model.state_dict(), cql_path)

    with torch.no_grad():
        q = model(X_t)
        expert_q = q.gather(1, A_t.unsqueeze(1)).mean().item()
        random_idx = torch.randint(0, len(ACTION_SPACE_7), (len(A_t),))
        random_q = q.gather(1, random_idx.unsqueeze(1)).mean().item()

    return {
        "cql_checkpoint": str(cql_path),
        "mean_q_expert": float(expert_q),
        "mean_q_random": float(random_q),
    }


def ppo_finetune_stub(cql_checkpoint: Path, out_dir: Path, total_steps: int = 10000) -> Dict[str, float]:
    """
    Lightweight PPO-style loop stub using policy logits from CQL initialization.
    Keeps runtime practical while preserving BC -> CQL -> PPO sequence structure.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for PPO finetuning.")

    out_dir.mkdir(parents=True, exist_ok=True)

    policy = PolicyNet(obs_dim=5, n_actions=len(ACTION_SPACE_7))
    policy.load_state_dict(torch.load(cql_checkpoint, map_location="cpu"), strict=False)

    env = CustomerSupportPOMDP()
    entropy_values = []
    rewards = []

    for _ in range(total_steps):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            x = torch.tensor(
                [
                    obs["belief_frustration"],
                    obs["belief_sentiment"],
                    float(obs["turn_index"]) / 12.0,
                    obs["belief_escalation_risk"],
                    float(TIER_VALUE_WEIGHT.get(str(obs.get("tier", "Pro")), 0.4)),
                ],
                dtype=torch.float32,
            ).unsqueeze(0)
            with torch.no_grad():
                logits = policy(x)
                probs = torch.softmax(logits, dim=1).squeeze(0)
            action = int(torch.multinomial(probs, num_samples=1).item())
            entropy_values.append(float(-(probs * torch.log(probs + 1e-9)).sum().item()))
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)

    ppo_path = out_dir / "ppo_policy_stub.pt"
    torch.save(policy.state_dict(), ppo_path)

    result = {
        "ppo_checkpoint": str(ppo_path),
        "mean_episode_reward": float(np.mean(rewards) if rewards else 0.0),
        "mean_entropy": float(np.mean(entropy_values) if entropy_values else 0.0),
        "episodes": int(len(rewards)),
    }
    return result


def run_training_pipeline(
    labeled_df: pd.DataFrame,
    reward_weights_path: Path,
    out_dir: Path,
    bc_epochs: int = 40,
    cql_steps: int = 200,
    ppo_steps: int = 500,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    bc = behavior_cloning_train(labeled_df, out_dir / "bc", epochs=bc_epochs)
    print(f"[training] BC done in {time.time() - t0:.1f}s")

    t1 = time.time()
    cql = cql_lite_train(
        labeled_df,
        reward_weights_path=reward_weights_path,
        bc_checkpoint=Path(bc["bc_checkpoint"]),
        out_dir=out_dir / "cql",
        gradient_steps=cql_steps,
    )
    print(f"[training] CQL done in {time.time() - t1:.1f}s")

    t2 = time.time()
    ppo = ppo_finetune_stub(Path(cql["cql_checkpoint"]), out_dir=out_dir / "ppo", total_steps=ppo_steps)
    print(f"[training] PPO done in {time.time() - t2:.1f}s")

    summary = {"bc": bc, "cql": cql, "ppo": ppo}
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
