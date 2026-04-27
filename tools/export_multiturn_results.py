from __future__ import annotations

import argparse
import json
import subprocess
import sys
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


def _write_visualize_script(*, plots_dir: Path) -> Path:
    """Write a runnable visualization script into the given plots folder.

    The script reads `../logs/full_pipeline_report.json` and regenerates the plots.
    """

    script_path = plots_dir / "visualize.py"
    script_path.write_text(
        """from __future__ import annotations

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
""",
        encoding="utf-8",
    )
    return script_path


def export_run(artifacts_root: Path, run_subdir: str, out_dir: Path) -> None:
    run_root = artifacts_root / run_subdir
    report_path = run_root / "full_pipeline_report.json"

    # Desired layout:
    #   output/<method_name>/*.json  (log files)
    #   output/<method_name>/plots/visualize.py (+ generated pngs)
    #   output/<method_name>/models/*.zip
    plots_dir = out_dir / "plots"
    models_dir = out_dir / "models"

    _mkdir_clean(out_dir)
    _mkdir_clean(plots_dir)
    _mkdir_clean(models_dir)

    # Copy key logs / reports
    _copy_if_exists(report_path, out_dir / report_path.name)
    _copy_if_exists(run_root / "training_summary.json", out_dir / "training_summary.json")
    _copy_if_exists(run_root / "training_log.json", out_dir / "training_log.json")
    _copy_if_exists(run_root / "validation_report_pretrain.json", out_dir / "validation_report_pretrain.json")
    _copy_if_exists(run_root / "demo_rollouts.json", out_dir / "demo_rollouts.json")

    # Copy model artifacts if they exist
    _copy_if_exists(run_root / "models" / "best_model.zip", models_dir / "best_model.zip")
    _copy_if_exists(run_root / "models" / "final_model.zip", models_dir / "final_model.zip")

    # Write + run visualization script as the final step
    script_path = _write_visualize_script(plots_dir=plots_dir)
    subprocess.run([sys.executable, str(script_path)], check=False)


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
