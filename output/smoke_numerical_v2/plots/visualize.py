from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _bar(ax, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)


def main() -> int:
    plots_dir = Path(__file__).resolve().parent
    method_dir = (plots_dir / "..").resolve()
    report_path = method_dir / "full_pipeline_report.json"
    report = _safe_read_json(report_path)

    if not report:
        print(f"No report found at: {report_path}")
        return 0

    evaluation = (report.get("evaluation") or {}) if isinstance(report, dict) else {}
    leaderboard = evaluation.get("leaderboard", []) if isinstance(evaluation, dict) else []

    # Persist a simple table for downstream parsing.
    (method_dir / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")

    # Plot 1: reward leaderboard
    if isinstance(leaderboard, list) and leaderboard:
        labels = [str(row.get("policy", "")) for row in leaderboard]
        values = [float(row.get("mean_reward", 0.0)) for row in leaderboard]
        fig, ax = plt.subplots(figsize=(10, 4))
        _bar(ax, labels, values, "Mean Reward (PPO vs baselines)", "mean_reward")
        fig.tight_layout()
        fig.savefig(plots_dir / "mean_reward_leaderboard.png", dpi=160)
        plt.close(fig)

    # Plot 2: terminal outcomes
    ppo = evaluation.get("ppo", {}) if isinstance(evaluation, dict) else {}
    baselines = evaluation.get("baselines", {}) if isinstance(evaluation, dict) else {}
    doc = baselines.get("document_guided", {}) if isinstance(baselines, dict) else {}

    def _outcome_rates(block: dict[str, Any]) -> dict[str, float]:
        if not isinstance(block, dict):
            return {"success": 0.0, "escalation": 0.0, "dropout": 0.0, "timeout": 0.0}
        return {
            "success": float(block.get("resolution_rate", 0.0)),
            "escalation": float(block.get("escalation_rate", 0.0)),
            "dropout": float(block.get("dropout_rate", 0.0)),
            "timeout": float(block.get("timeout_rate", 0.0)),
        }

    ppo_rates = _outcome_rates(ppo)
    doc_rates = _outcome_rates(doc)

    labels = ["success", "escalation", "dropout", "timeout"]
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([i - 0.2 for i in x], [ppo_rates[k] for k in labels], width=0.4, label="PPO")
    ax.bar([i + 0.2 for i in x], [doc_rates[k] for k in labels], width=0.4, label="document_guided")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("rate")
    ax.set_title("Terminal Outcome Rates")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "terminal_outcomes.png", dpi=160)
    plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
