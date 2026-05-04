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

    # Plots 3-5: training curves from training_log.json
    training_log = _safe_read_json(method_dir / "training_log.json")
    metrics_history = training_log.get("metrics_history", []) if isinstance(training_log, dict) else []
    eval_history = training_log.get("eval_history", []) if isinstance(training_log, dict) else []

    if isinstance(metrics_history, list) and len(metrics_history) > 1:
        steps = [int(r.get("timesteps", 0)) for r in metrics_history]
        rewards = [float(r.get("mean_reward_100ep", 0.0)) for r in metrics_history]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, rewards, linewidth=1.5)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Mean Reward (100-ep rolling)")
        ax.set_title("Training Reward Curve")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "reward_curve.png", dpi=160)
        plt.close(fig)

        res_rates = [float(r.get("resolution_rate", 0.0)) for r in metrics_history]
        esc_rates = [float(r.get("escalation_rate", 0.0)) for r in metrics_history]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, res_rates, label="resolution", linewidth=1.5)
        ax.plot(steps, esc_rates, label="escalation", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Rate")
        ax.set_title("Resolution & Escalation Rate During Training")
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "resolution_curve.png", dpi=160)
        plt.close(fig)

    if isinstance(eval_history, list) and len(eval_history) > 1:
        eval_steps = [int(r.get("timesteps", 0)) for r in eval_history]
        eval_rewards = [float(r.get("mean_reward", 0.0)) for r in eval_history]
        eval_res = [float(r.get("resolution_rate", 0.0)) for r in eval_history]

        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax2 = ax1.twinx()
        ax1.plot(eval_steps, eval_rewards, color="steelblue", label="eval reward", linewidth=1.5)
        ax2.plot(eval_steps, eval_res, color="darkorange", label="eval resolution", linewidth=1.5, linestyle="--")
        ax1.set_xlabel("Timesteps")
        ax1.set_ylabel("Mean Reward", color="steelblue")
        ax2.set_ylabel("Resolution Rate", color="darkorange")
        ax1.set_title("Eval Reward & Resolution Rate")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "eval_curve.png", dpi=160)
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
