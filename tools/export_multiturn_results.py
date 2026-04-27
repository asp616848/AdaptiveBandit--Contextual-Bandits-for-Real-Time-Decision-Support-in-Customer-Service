from __future__ import annotations

import argparse
import json
import shutil
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


def _mkdir_clean(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _bar(ax, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)


def export_run(artifacts_root: Path, run_subdir: str, out_dir: Path) -> None:
    run_root = artifacts_root / run_subdir
    report_path = run_root / "full_pipeline_report.json"

    report = _safe_read_json(report_path)

    _mkdir_clean(out_dir)
    _mkdir_clean(out_dir / "plots")
    _mkdir_clean(out_dir / "reports")
    _mkdir_clean(out_dir / "models")

    # Copy key reports (JSON)
    _copy_if_exists(report_path, out_dir / "reports" / report_path.name)
    _copy_if_exists(run_root / "training_summary.json", out_dir / "reports" / "training_summary.json")
    _copy_if_exists(run_root / "training_log.json", out_dir / "reports" / "training_log.json")
    _copy_if_exists(run_root / "validation_report_pretrain.json", out_dir / "reports" / "validation_report_pretrain.json")
    _copy_if_exists(run_root / "demo_rollouts.json", out_dir / "reports" / "demo_rollouts.json")

    # Copy model artifacts if they exist
    _copy_if_exists(run_root / "models" / "best_model.zip", out_dir / "models" / "best_model.zip")
    _copy_if_exists(run_root / "models" / "final_model.zip", out_dir / "models" / "final_model.zip")

    evaluation = (report.get("evaluation") or {}) if isinstance(report, dict) else {}
    leaderboard = evaluation.get("leaderboard", []) if isinstance(evaluation, dict) else []

    # Write a small flat CSV-like table (JSONL) that is easy to parse
    leaderboard_path = out_dir / "reports" / "leaderboard.json"
    leaderboard_path.write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")

    # Plots
    ppo = evaluation.get("ppo", {}) if isinstance(evaluation, dict) else {}
    baselines = evaluation.get("baselines", {}) if isinstance(evaluation, dict) else {}

    # Reward leaderboard bar chart
    if isinstance(leaderboard, list) and leaderboard:
        labels = [str(row.get("policy", "")) for row in leaderboard]
        values = [float(row.get("mean_reward", 0.0)) for row in leaderboard]
        fig, ax = plt.subplots(figsize=(10, 4))
        _bar(ax, labels, values, "Mean Reward (PPO vs baselines)", "mean_reward")
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / "mean_reward_leaderboard.png", dpi=160)
        plt.close(fig)

    # Terminal outcomes comparison (PPO vs document_guided)
    def _outcome_rates(block: dict[str, Any]) -> dict[str, float]:
        if not isinstance(block, dict):
            return {"success": 0.0, "escalation": 0.0, "dropout": 0.0, "timeout": 0.0}
        return {
            "success": float(block.get("resolution_rate", 0.0)),
            "escalation": float(block.get("escalation_rate", 0.0)),
            "dropout": float(block.get("dropout_rate", 0.0)),
            "timeout": float(block.get("timeout_rate", 0.0)),
        }

    doc = baselines.get("document_guided", {}) if isinstance(baselines, dict) else {}
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
    fig.savefig(out_dir / "plots" / "terminal_outcomes.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-root",
        type=str,
        default=str(Path("Multi-Turn RL") / "Simulation_4" / "artifacts"),
        help="Path to Simulation_4/artifacts",
    )
    parser.add_argument("--run-subdir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    export_run(
        artifacts_root=Path(args.artifacts_root),
        run_subdir=str(args.run_subdir),
        out_dir=Path(args.out_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
